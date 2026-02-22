"""Wake word / hotkey detection for hands-free NOVA activation.

Uses pynput keyboard hotkey listener as a cross-platform fallback
(pvporcupine is incompatible with Python 3.14).
"""

import asyncio
import io
import logging
import math
import struct
import wave

from nova.config import get_config

logger = logging.getLogger(__name__)


def generate_beep(
    frequency: float = 440.0,
    duration: float = 0.2,
    sample_rate: int = 16000,
    volume: float = 0.3,
) -> bytes:
    """Generate a short sine-wave activation beep as WAV bytes.

    Args:
        frequency: Tone frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Audio sample rate.
        volume: Volume multiplier (0.0–1.0).

    Returns:
        WAV-encoded audio bytes.
    """
    num_samples = int(sample_rate * duration)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Apply a short fade-in/out to avoid clicks
        envelope = 1.0
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        if i < fade_samples:
            envelope = i / fade_samples
        elif i > num_samples - fade_samples:
            envelope = (num_samples - i) / fade_samples
        value = volume * envelope * math.sin(2.0 * math.pi * frequency * t)
        samples.append(int(value * 32767))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


class HotkeyWakeWordDetector:
    """Listens for a keyboard hotkey to activate NOVA.

    Uses pynput.keyboard.GlobalHotKeys in a background thread.
    When the configured hotkey is pressed, an asyncio Event is set
    and a short activation beep is played.
    """

    def __init__(self) -> None:
        config = get_config()
        self._hotkey = config.wake_word_hotkey
        self._listener = None
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._beep_bytes = generate_beep()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the hotkey listener in a background thread.

        Args:
            loop: The running asyncio event loop for thread-safe event signalling.
        """
        from pynput.keyboard import GlobalHotKeys

        self._loop = loop
        self._event = asyncio.Event()

        def _on_activate():
            logger.info("Wake word hotkey detected (%s)", self._hotkey)
            if self._loop and self._event:
                self._loop.call_soon_threadsafe(self._event.set)

        self._listener = GlobalHotKeys({self._hotkey: _on_activate})
        self._listener.daemon = True
        self._listener.start()
        logger.info("Hotkey listener started — press %s to activate", self._hotkey)

    async def wait_for_activation(self) -> None:
        """Wait until the hotkey is pressed.

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
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("Hotkey listener stopped")


if __name__ == "__main__":
    async def main() -> None:
        print("=== Wake Word (Hotkey) Test ===")
        detector = HotkeyWakeWordDetector()
        loop = asyncio.get_event_loop()
        detector.start(loop)

        print(f"Press {detector._hotkey} to activate (Ctrl+C to exit)")
        try:
            while True:
                await detector.wait_for_activation()
                print("✅ Activated! (listening would start here)")
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            detector.stop()

    asyncio.run(main())
