"""DeepTalk continuous conversation session.

Manages the loop: listen → STT → (exit check) → LLM → TTS → listen …
with barge-in detection during TTS playback.
"""

import asyncio
import logging
import threading

from rich.console import Console

from nova.config import get_config

logger = logging.getLogger(__name__)
console = Console()

# Phrases that exit DeepTalk mode (checked against STT output)
EXIT_PHRASES = [
    "nova selesai",
    "selesai",
    "stop",
    "keluar",
    "exit deep talk",
    "exit deeptalk",
    "mode biasa",
    "berhenti",
]

# Phrases that trigger DeepTalk entry from normal mode
DEEPTALK_TRIGGERS = [
    "mode deeptalk",
    "mode deep talk",
    "mulai deeptalk",
    "mulai deep talk",
    "deep talk mode",
    "deeptalk mode",
    "aktifkan deeptalk",
    "start deep talk",
    "start deeptalk",
]


def is_exit_phrase(text: str) -> bool:
    """Check if transcribed text is a DeepTalk exit command.

    Uses word-boundary matching to avoid false positives like
    "selesaikan" matching the "selesai" exit phrase.

    Args:
        text: The STT transcript to check.

    Returns:
        True if the text is an exit command.
    """
    normalized = text.strip().lower()
    if not normalized:
        return False
    for phrase in EXIT_PHRASES:
        idx = normalized.find(phrase)
        if idx == -1:
            continue
        # Word boundary before
        if idx > 0 and normalized[idx - 1].isalpha():
            continue
        # Word boundary after
        end = idx + len(phrase)
        if end < len(normalized) and normalized[end].isalpha():
            continue
        return True
    return False


def is_deeptalk_trigger(text: str) -> bool:
    """Check if transcript is a command to enter DeepTalk mode.

    Args:
        text: The STT transcript to check.

    Returns:
        True if the text triggers DeepTalk entry.
    """
    normalized = text.strip().lower()
    return any(trigger in normalized for trigger in DEEPTALK_TRIGGERS)


class DeepTalkSession:
    """Manages a continuous conversation session without wake words.

    The loop:
    1. Listen for speech (Silero VAD — ignores keyboard/ambient noise)
    2. Transcribe (STT)
    3. Check for exit phrase → end session
    4. Start barge-in monitoring
    5. Send to LLM → stream TTS
    6. If barge-in → TTS stops, loop continues
    7. After TTS finishes, back to step 1
    """

    def __init__(self, orchestrator, barge_in_detector) -> None:
        self._orchestrator = orchestrator
        self._barge_in = barge_in_detector
        self._active = False
        self._config = get_config()

    @property
    def is_active(self) -> bool:
        return self._active

    async def start(self) -> None:
        """Enter DeepTalk mode, announce, and run the conversation loop."""
        self._active = True
        logger.info("DeepTalk session started")

        console.print(
            "\n[bold magenta]🔊 DeepTalk mode[/] — "
            "bicara langsung tanpa wake word. "
            'Katakan [bold]"selesai"[/] untuk keluar.\n'
        )

        await self._orchestrator.speak(
            "Mode DeepTalk aktif. Silakan bicara kapan saja, Tuan.",
            language="id",
        )
        await self._loop()

    async def _loop(self) -> None:
        """Main DeepTalk conversation loop."""
        while self._active:
            try:
                # 1. Capture speech via Silero VAD (ignores keyboard/noise)
                audio_data = await self._capture_audio()
                if audio_data is None or len(audio_data) <= 44:
                    continue

                # 2. Transcribe
                text = await self._transcribe(audio_data)
                if not text or not text.strip():
                    continue

                text = text.strip()
                logger.info("DeepTalk heard: '%s'", text)

                # 3. Check exit phrase BEFORE sending to LLM
                if is_exit_phrase(text):
                    await self._exit()
                    return

                # 4. Process: LLM → streaming TTS, with barge-in monitoring
                self._orchestrator.reset_speaking()

                self._barge_in.on_interrupt = self._handle_barge_in
                self._barge_in.start_monitoring()

                try:
                    response = await self._orchestrator.handle_interaction(text)
                    console.print(f"[bold white]You:[/] {text}")
                    if response:
                        console.print(f"[bold cyan]Nova:[/] {response}\n")
                finally:
                    self._barge_in.stop_monitoring()

            except asyncio.CancelledError:
                self._barge_in.stop_capture()
                logger.info("DeepTalk session cancelled")
                self._active = False
                return
            except Exception:
                logger.exception("DeepTalk loop error")
                await asyncio.sleep(0.5)

    async def _capture_audio(self) -> bytes | None:
        """Capture speech using Silero VAD in a dedicated thread.

        Uses a real ``threading.Thread`` instead of the asyncio executor
        pool because Windows WASAPI/COM audio requires proper per-thread
        initialisation that pool threads don't guarantee.
        """
        result: bytes | None = None
        error: BaseException | None = None
        done = asyncio.Event()
        loop = asyncio.get_event_loop()

        def _run() -> None:
            nonlocal result, error
            try:
                result = self._barge_in.capture_speech(
                    silence_duration=self._config.deeptalk_silence_duration,
                )
            except OSError as exc:
                error = exc
            except Exception as exc:
                error = exc
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(
            target=_run, name="deeptalk-capture", daemon=True,
        )
        thread.start()
        await done.wait()
        thread.join(timeout=1.0)

        if isinstance(error, OSError):
            logger.error("DeepTalk: audio device error")
            self._active = False
            return None
        if error is not None:
            logger.warning("DeepTalk: capture error: %s", error)
            return None
        return result

    async def _transcribe(self, wav_bytes: bytes) -> str | None:
        """Transcribe audio via the STT router."""
        try:
            return await self._orchestrator._stt_router.execute(
                "transcribe", wav_bytes,
            )
        except Exception:
            logger.warning("DeepTalk: STT failed", exc_info=True)
            return None

    def _handle_barge_in(self) -> None:
        """Called from the barge-in thread when user speech is detected."""
        self._orchestrator.stop_speaking()
        logger.info("DeepTalk: barge-in — TTS stopped")

    async def _exit(self) -> None:
        """Exit DeepTalk mode with an announcement."""
        self._active = False
        logger.info("DeepTalk session ended")
        await self._orchestrator.speak("Mode DeepTalk selesai.", language="id")
        console.print("[bold magenta]DeepTalk mode ended[/] — kembali ke mode biasa.\n")
