"""Audio capture with energy-based Voice Activity Detection (VAD)."""

import asyncio
import enum
import io
import logging
import wave

import numpy as np
import sounddevice as sd

from nova.config import get_config

logger = logging.getLogger(__name__)


class _State(enum.Enum):
    WAITING = "waiting"
    RECORDING = "recording"
    DONE = "done"


class AudioCapture:
    """Records audio from the microphone with automatic silence detection.

    Uses a state machine: WAITING -> RECORDING -> DONE.
    WAITING: listens for audio energy above the threshold.
    RECORDING: captures audio, stops after silence_duration of silence.
    DONE: returns the captured audio as WAV bytes.
    """

    def __init__(self) -> None:
        config = get_config()
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.silence_threshold = config.silence_threshold
        self.silence_duration = config.silence_duration
        self.max_recording_seconds = config.max_recording_seconds
        self.min_recording_seconds = 1.5  # Pad short clips to avoid hallucinations
        # Process audio in 100ms chunks
        self._chunk_duration = 0.1
        self._chunk_samples = int(self.sample_rate * self._chunk_duration)

    @staticmethod
    def _rms(audio_chunk: np.ndarray) -> float:
        """Calculate Root Mean Square energy of an audio chunk."""
        float_data = audio_chunk.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(float_data ** 2)))

    async def capture(self) -> bytes:
        """Record audio from the microphone until speech ends.

        Returns:
            WAV-encoded audio bytes (16kHz, mono, 16-bit PCM).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_sync)

    def _capture_sync(self) -> bytes:
        """Synchronous capture implementation (runs in executor)."""
        state = _State.WAITING
        recorded_chunks: list[np.ndarray] = []
        silence_samples = 0
        max_samples = int(self.max_recording_seconds * self.sample_rate)
        silence_samples_threshold = int(self.silence_duration * self.sample_rate)
        total_recorded = 0

        logger.info("AudioCapture: listening (threshold=%.4f)...", self.silence_threshold)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self._chunk_samples,
        ) as stream:
            while state != _State.DONE:
                chunk, overflowed = stream.read(self._chunk_samples)
                if overflowed:
                    logger.debug("AudioCapture: input overflow")

                energy = self._rms(chunk)

                if state == _State.WAITING:
                    if energy > self.silence_threshold:
                        state = _State.RECORDING
                        recorded_chunks.append(chunk.copy())
                        total_recorded += len(chunk)
                        silence_samples = 0
                        logger.info(
                            "AudioCapture: speech detected (energy=%.4f), recording...",
                            energy,
                        )

                elif state == _State.RECORDING:
                    recorded_chunks.append(chunk.copy())
                    total_recorded += len(chunk)

                    if energy < self.silence_threshold:
                        silence_samples += len(chunk)
                        if silence_samples >= silence_samples_threshold:
                            state = _State.DONE
                            logger.info("AudioCapture: silence detected, stopping")
                    else:
                        silence_samples = 0

                    if total_recorded >= max_samples:
                        state = _State.DONE
                        logger.info("AudioCapture: max recording time reached")

        if not recorded_chunks:
            logger.warning("AudioCapture: no audio captured")
            return self._empty_wav()

        audio_data = np.concatenate(recorded_chunks)
        duration = len(audio_data) / self.sample_rate

        # Pad short clips with silence to reach minimum duration
        min_samples = int(self.min_recording_seconds * self.sample_rate)
        if len(audio_data) < min_samples:
            padding = np.zeros(min_samples - len(audio_data), dtype=audio_data.dtype)
            audio_data = np.concatenate([audio_data, padding])
            logger.info(
                "AudioCapture: padded %.1fs → %.1fs (min duration)",
                duration, self.min_recording_seconds,
            )

        logger.info("AudioCapture: captured %.1fs of audio", duration)

        return self._to_wav(audio_data)

    def _to_wav(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return buffer.getvalue()

    def _empty_wav(self) -> bytes:
        """Return a minimal empty WAV file."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"")
        return buffer.getvalue()


if __name__ == "__main__":
    import os

    async def main() -> None:
        capture = AudioCapture()

        print("=== Audio Capture Test ===")
        print(f"Sample rate: {capture.sample_rate} Hz")
        print(f"Silence threshold: {capture.silence_threshold}")
        print(f"Silence duration: {capture.silence_duration}s")
        print(f"Max recording: {capture.max_recording_seconds}s")
        print()
        print("Listening... speak into your microphone.")
        print("(Recording will stop after silence is detected)")
        print()

        wav_bytes = await capture.capture()

        duration = (len(wav_bytes) - 44) / (capture.sample_rate * 2)  # 44-byte WAV header
        print(f"\nCaptured: {duration:.1f}s, {len(wav_bytes):,} bytes")

        # Save to test.wav
        output_path = "test.wav"
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        file_size = os.path.getsize(output_path)
        print(f"Saved to: {output_path} ({file_size:,} bytes)")
        print(f"\nPlayback: ffplay -nodisp -autoexit {output_path}")

    asyncio.run(main())
