"""OpenWakeWord-based wake word detector for hands-free NOVA activation.

Uses the openwakeword library with a custom ONNX model to continuously
listen for the wake phrase (e.g. "hey nova") via microphone streaming.
Falls back to HotkeyWakeWordDetector if openwakeword cannot be loaded.
"""

import asyncio
import logging
import threading

import sounddevice as sd

from nova.audio.wake_word import generate_beep
from nova.config import get_config

logger = logging.getLogger(__name__)

# Frame size: 80ms at 16kHz = 1280 samples (openwakeword recommended)
_FRAME_SAMPLES = 1280
_SAMPLE_RATE = 16000


class OpenWakeWordDetector:
    """Listens for a wake word using openwakeword with a custom ONNX model.

    Opens a continuous InputStream and reads 80ms frames in a dedicated
    thread, feeding each frame to openwakeword for prediction.  When the
    score exceeds the configured threshold, an asyncio Event is set and
    an activation beep is played.
    """

    def __init__(self) -> None:
        config = get_config()
        self._model_path: str = config.wake_word_model_path
        self._threshold: float = config.wake_word_threshold
        self._vad_threshold: float = config.wake_word_vad_threshold
        self._model = None
        self._stream: sd.InputStream | None = None
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._beep_bytes = generate_beep()
        # Cooldown: ignore predictions for N frames after activation
        self._cooldown_frames = 0
        self._cooldown_total = 20  # ~1.6s at 80ms/frame

    def _load_model(self) -> None:
        """Load the openwakeword model from disk."""
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[self._model_path],
            inference_framework="onnx",
            vad_threshold=self._vad_threshold,
        )
        model_names = list(self._model.models.keys())
        logger.info(
            "OpenWakeWord model loaded: %s (threshold=%.2f, vad=%.2f)",
            model_names,
            self._threshold,
            self._vad_threshold,
        )

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the wake word listener with continuous mic streaming.

        Args:
            loop: The running asyncio event loop for thread-safe signalling.
        """
        self._loop = loop
        self._event = asyncio.Event()

        # Load model (may raise if file missing or openwakeword broken)
        self._load_model()

        self._running = True

        # Open the stream — no callback; we read from it in a thread
        self._stream = sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_FRAME_SAMPLES,
        )
        self._stream.start()

        # Spawn a dedicated reader thread
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="oww-reader",
            daemon=True,
        )
        self._thread.start()
        logger.info("OpenWakeWord detector started — listening for wake word")

    def _reader_loop(self) -> None:
        """Continuously read audio frames and run prediction.

        Runs in a dedicated thread.  stream.read() blocks until a full
        frame (1280 samples) is available, guaranteeing continuous audio
        with no gaps between frames.
        """
        while self._running and self._stream is not None:
            try:
                audio, overflowed = self._stream.read(_FRAME_SAMPLES)
            except Exception:
                if self._running:
                    logger.debug("stream.read error", exc_info=True)
                break

            if overflowed:
                logger.debug("Audio input overflowed")

            # Cooldown after activation to prevent re-triggering
            if self._cooldown_frames > 0:
                self._cooldown_frames -= 1
                continue

            # Flatten to 1-D int16 array
            audio_frame = audio[:, 0]

            # Run prediction
            try:
                prediction = self._model.predict(audio_frame)
            except Exception:
                logger.debug("OpenWakeWord predict error", exc_info=True)
                continue

            # Check if any model score exceeds threshold
            for model_name, score in prediction.items():
                if score > self._threshold:
                    logger.info(
                        "Wake word detected! model=%s score=%.3f",
                        model_name,
                        score,
                    )
                    # Enter cooldown to prevent rapid re-triggering
                    self._cooldown_frames = self._cooldown_total
                    # Reset prediction buffer
                    self._model.reset()

                    # Signal the event loop
                    with self._lock:
                        if self._loop and self._event:
                            self._loop.call_soon_threadsafe(self._event.set)
                    break

    async def wait_for_activation(self) -> None:
        """Wait until the wake word is detected.

        Plays an activation beep upon detection.
        """
        if self._event is None:
            raise RuntimeError("Detector not started — call start() first")

        self._event.clear()
        await self._event.wait()

        # Play activation beep
        try:
            from nova.audio.playback import play_audio

            await play_audio(self._beep_bytes)
        except Exception:
            logger.debug("Could not play activation beep", exc_info=True)

    def stop(self) -> None:
        """Stop the wake word listener and release resources."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("Error stopping audio stream", exc_info=True)
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._model = None
        logger.info("OpenWakeWord detector stopped")

    @property
    def model_name(self) -> str:
        """Return the loaded model path for display."""
        return self._model_path


if __name__ == "__main__":

    async def main() -> None:
        print("=== OpenWakeWord Detector Test ===")
        detector = OpenWakeWordDetector()
        loop = asyncio.get_event_loop()

        try:
            detector.start(loop)
        except Exception as e:
            print(f"Failed to start detector: {e}")
            return

        print(f"Model: {detector.model_name}")
        print("Say the wake word to activate (Ctrl+C to exit)\n")

        try:
            while True:
                await detector.wait_for_activation()
                print("Wake word detected! (listening would start here)")
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            detector.stop()

    asyncio.run(main())
