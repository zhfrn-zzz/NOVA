"""Silero VAD integration for DeepTalk — barge-in detection and speech capture.

Uses the Silero VAD ONNX model (~2 MB) for:
  1. **Speech capture** — VAD-gated mic recording that only activates on
     real speech, ignoring keyboard noise and ambient sounds.
  2. **Barge-in detection** — monitors mic during TTS to detect user
     interruption.

The ONNX model is loaded via ``onnxruntime`` (already installed for
openWakeWord).
"""

import io
import logging
import os
import threading
import urllib.request
import wave
from collections import deque
from typing import Callable

import numpy as np
import onnxruntime
import sounddevice as sd

from nova.config import get_config

logger = logging.getLogger(__name__)

_VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz


def _ensure_vad_model(path: str) -> str:
    """Download Silero VAD ONNX model if not present."""
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    logger.info("Downloading Silero VAD model to %s …", path)
    urllib.request.urlretrieve(_VAD_MODEL_URL, path)
    logger.info("Silero VAD model downloaded (%d KB)", os.path.getsize(path) // 1024)
    return path


class BargeInDetector:
    """Monitors microphone during TTS playback to detect user interruption.

    Uses Silero VAD (Voice Activity Detection) neural network to distinguish
    real human speech from TTS echo, background noise, and other sounds.
    The ONNX model runs via ``onnxruntime`` (~1 ms inference per 32 ms chunk).

    The detector auto-detects the Silero VAD model version (v4 or v5) from
    the ONNX input names and adapts its state management accordingly.
    """

    def __init__(self) -> None:
        config = get_config()
        model_path = _ensure_vad_model(config.deeptalk_vad_model_path)

        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Detect model version from input names
        input_names = {inp.name for inp in self._session.get_inputs()}
        self._v5 = "state" in input_names

        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

        self._vad_threshold: float = config.deeptalk_vad_threshold
        self._capture_threshold: float = config.deeptalk_capture_vad_threshold
        self._min_speech_duration: float = config.deeptalk_min_speech_duration

        # v6 context window: the model requires the last 64 samples from
        # the previous chunk prepended to each new 512-sample input.
        _ctx_size = 64 if SAMPLE_RATE == 16000 else 32
        self._context_size: int = _ctx_size
        self._context = np.zeros((1, _ctx_size), dtype=np.float32)

        self._monitoring = False
        self._capturing = False
        self._thread: threading.Thread | None = None
        self.on_interrupt: Callable[[], None] | None = None

        logger.info(
            "BargeInDetector loaded (v%s, barge_th=%.2f, capture_th=%.2f, "
            "min_dur=%.2fs)",
            "5" if self._v5 else "4",
            self._vad_threshold,
            self._capture_threshold,
            self._min_speech_duration,
        )

    def _reset_state(self) -> None:
        """Reset VAD internal state and context for a new session."""
        if self._v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def _predict(self, audio_chunk: np.ndarray) -> float:
        """Run VAD inference on a single float32 audio chunk.

        Silero VAD v6 expects each input to be context + new samples
        concatenated (576 samples = 64 context + 512 new at 16 kHz).

        Args:
            audio_chunk: 1-D float32 array, 512 samples at 16 kHz.

        Returns:
            Speech probability in [0, 1].
        """
        new_data = audio_chunk.reshape(1, -1).astype(np.float32)
        # Prepend context from previous chunk
        input_data = np.concatenate([self._context, new_data], axis=1)

        if self._v5:
            outputs = self._session.run(
                None,
                {
                    "input": input_data,
                    "state": self._state,
                    "sr": np.array(SAMPLE_RATE, dtype=np.int64),
                },
            )
            self._state = outputs[1]
        else:
            outputs = self._session.run(
                None,
                {
                    "input": input_data,
                    "h": self._h,
                    "c": self._c,
                    "sr": np.array(SAMPLE_RATE, dtype=np.int64),
                },
            )
            self._h = outputs[1]
            self._c = outputs[2]

        # Save trailing samples as context for next call
        self._context = input_data[:, -self._context_size:]

        return float(outputs[0].item())

    def capture_speech(
        self,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
    ) -> bytes | None:
        """Capture speech from the microphone using Silero VAD.

        Waits for the user to start speaking (VAD probability above
        threshold for ``_min_speech_duration``), then records until
        speech ends (probability below threshold for ``silence_duration``).

        Unlike energy-based AudioCapture, this correctly ignores keyboard
        noise, ambient sounds, and other non-speech audio.

        Args:
            silence_duration: Seconds of silence after speech to stop.
            max_duration: Maximum recording length in seconds.

        Returns:
            WAV bytes (16 kHz, mono, 16-bit PCM), or None if cancelled
            or no speech detected.
        """
        chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE
        pre_buf_len = int(0.3 / chunk_dur)  # ~300 ms lookback
        # Diagnostic: track peak probability + RMS for debugging
        diag_interval = int(3.0 / chunk_dur)  # log every ~3 s
        diag_counter = 0
        diag_max_prob = 0.0
        diag_max_rms = 0.0

        self._reset_state()
        self._capturing = True

        pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buf_len)
        recorded: list[np.ndarray] = []
        speech_started = False
        consecutive_speech = 0.0
        consecutive_silence = 0.0
        detection_chunks: list[np.ndarray] = []
        record_time = 0.0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
            ) as stream:
                logger.info(
                    "DeepTalk VAD: listening (threshold=%.2f)...",
                    self._capture_threshold,
                )

                while self._capturing:
                    try:
                        audio_int16, _ = stream.read(CHUNK_SAMPLES)
                    except Exception:
                        if self._capturing:
                            logger.debug("VAD capture read error", exc_info=True)
                        break

                    chunk_f32 = audio_int16[:, 0].astype(np.float32) / 32768.0
                    prob = self._predict(chunk_f32)
                    rms = float(np.sqrt(np.mean(chunk_f32 ** 2)))

                    # Periodic diagnostic logging
                    diag_max_prob = max(diag_max_prob, prob)
                    diag_max_rms = max(diag_max_rms, rms)
                    diag_counter += 1
                    if diag_counter >= diag_interval:
                        if not speech_started:
                            logger.debug(
                                "DeepTalk VAD: waiting (peak_prob=%.3f, peak_rms=%.4f)",
                                diag_max_prob,
                                diag_max_rms,
                            )
                        diag_counter = 0
                        diag_max_prob = 0.0
                        diag_max_rms = 0.0

                    if not speech_started:
                        if prob > self._capture_threshold:
                            consecutive_speech += chunk_dur
                            detection_chunks.append(audio_int16.copy())
                            if consecutive_speech >= self._min_speech_duration:
                                speech_started = True
                                recorded.extend(pre_buffer)
                                recorded.extend(detection_chunks)
                                detection_chunks.clear()
                                logger.info(
                                    "DeepTalk VAD: speech detected (prob=%.3f)",
                                    prob,
                                )
                        else:
                            consecutive_speech = 0.0
                            detection_chunks.clear()
                            pre_buffer.append(audio_int16.copy())
                    else:
                        recorded.append(audio_int16.copy())
                        record_time += chunk_dur

                        if record_time >= max_duration:
                            logger.info("DeepTalk VAD: max duration reached")
                            break

                        if prob < self._capture_threshold:
                            consecutive_silence += chunk_dur
                            if consecutive_silence >= silence_duration:
                                break
                        else:
                            consecutive_silence = 0.0
        except Exception:
            logger.warning("DeepTalk VAD: capture stream error", exc_info=True)
            self._reset_state()
            self._capturing = False
            return None

        self._reset_state()
        self._capturing = False

        if not recorded:
            return None

        audio_data = np.concatenate(recorded)
        duration = len(audio_data) / SAMPLE_RATE
        logger.info("DeepTalk VAD: captured %.1fs of speech", duration)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def stop_capture(self) -> None:
        """Cancel an in-progress capture_speech call (thread-safe)."""
        self._capturing = False

    def start_monitoring(self) -> None:
        """Start monitoring the microphone for barge-in."""
        if self._monitoring:
            return
        self._monitoring = True
        self._reset_state()
        self._thread = threading.Thread(
            target=self._monitor_loop, name="barge-in", daemon=True,
        )
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring and join the background thread."""
        self._monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Background thread: read mic chunks, run VAD, trigger on speech."""
        speech_duration = 0.0
        chunk_duration = CHUNK_SAMPLES / SAMPLE_RATE

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SAMPLES,
            ) as stream:
                while self._monitoring:
                    try:
                        audio, _overflow = stream.read(CHUNK_SAMPLES)
                    except Exception:
                        if self._monitoring:
                            logger.debug("Barge-in stream.read error", exc_info=True)
                        break

                    chunk = audio[:, 0].astype(np.float32) / 32768.0
                    speech_prob = self._predict(chunk)

                    if speech_prob > self._vad_threshold:
                        speech_duration += chunk_duration
                        if speech_duration >= self._min_speech_duration:
                            logger.info(
                                "Barge-in triggered: prob=%.3f dur=%.2fs",
                                speech_prob, speech_duration,
                            )
                            if self.on_interrupt:
                                self.on_interrupt()
                            self._monitoring = False
                            return
                    else:
                        speech_duration = 0.0
        except Exception:
            logger.warning("Barge-in monitor stream error", exc_info=True)
