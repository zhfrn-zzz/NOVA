"""Streaming TTS — split-and-stream approach to reduce perceived latency.

Instead of waiting for the full response to be synthesized, this module:
1. Splits the LLM response into sentences.
2. Synthesizes the first sentence → starts playing immediately.
3. While playing, synthesizes the next sentence in the background.
4. Queues audio chunks for gapless back-to-back playback.

This dramatically reduces "time to first audio" from 8+ seconds to <2 seconds.
"""

import asyncio
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator

from nova.audio.playback import _find_player

logger = logging.getLogger(__name__)

# Regex patterns for stripping markdown before TTS
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_BULLET = re.compile(r'^[*\-•]\s+', re.MULTILINE)
_MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_CODE = re.compile(r'`(.+?)`')


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for TTS output.

    Strips bold, italic, bullets, headers, and inline code so that
    TTS engines don't read out asterisks or hashes.

    Args:
        text: Raw text possibly containing markdown.

    Returns:
        Plain text suitable for speech synthesis.
    """
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_BULLET.sub('', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_CODE.sub(r'\1', text)
    return text

# Abbreviations that end with a period but are NOT sentence boundaries.
# Covers Indonesian and English common abbreviations.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "jr", "sr", "vs", "etc", "inc", "ltd",
    "dll", "dsb", "dkk", "spt", "yth", "no", "vol", "hal", "tel", "fax",
}

# Regex: split on sentence-ending punctuation (. ! ?) followed by whitespace,
# but keep the punctuation attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into natural sentences for incremental TTS synthesis.

    Handles:
    - Standard sentence endings (. ! ?)
    - Abbreviations (dr., dll., etc.) — not treated as sentence breaks
    - Numbers with decimals (3.14) — not treated as sentence breaks
    - Very short fragments get merged with the next sentence

    Args:
        text: The full text to split.

    Returns:
        List of sentence strings, each suitable for independent TTS synthesis.
    """
    text = text.strip()
    if not text:
        return []

    # First pass: split on sentence-ending punctuation + whitespace
    raw_parts = _SENTENCE_SPLIT_RE.split(text)

    # Second pass: merge fragments that were split on abbreviations or are too short
    sentences: list[str] = []
    buffer = ""

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if buffer:
            # Check if the buffer ended with an abbreviation (not a real sentence break)
            last_word = buffer.rstrip(".!?").rsplit(None, 1)[-1].lower()
            if last_word in _ABBREVIATIONS:
                # Abbreviation — merge with this part
                buffer = buffer + " " + part
                continue

            # Check if buffer ended with a digit + period (decimal number like "3.14")
            if buffer.rstrip().endswith(".") and len(buffer) >= 2:
                char_before_dot = buffer.rstrip()[-2]
                if char_before_dot.isdigit():
                    buffer = buffer + " " + part
                    continue

            # Buffer is a real sentence — flush it if it's long enough
            if len(buffer) >= 40:
                sentences.append(buffer)
                buffer = part
            else:
                # Too short (< 40 chars) — merge with next for natural speech
                buffer = buffer + " " + part
        else:
            buffer = part

    # Flush remaining buffer
    if buffer:
        if sentences and len(buffer) < 40:
            # Very short trailing fragment — merge with last sentence
            sentences[-1] = sentences[-1] + " " + buffer
        else:
            sentences.append(buffer)

    return sentences


class StreamingTTSPlayer:
    """Overlapped TTS synthesis and playback for reduced latency.

    Uses an asyncio queue as a producer-consumer pipeline:
    - Producer: synthesizes sentences one by one, pushes audio bytes to queue
    - Consumer: plays audio chunks back-to-back from the queue

    While the consumer plays sentence N, the producer synthesizes sentence N+1.

    Supports mid-stream interruption via ``stop()`` for barge-in during
    DeepTalk mode.  The stop flag is a ``threading.Event`` so it can be
    safely set from background threads (e.g. the barge-in detector).
    """

    def __init__(self) -> None:
        self._stop_flag = threading.Event()
        self._playback_process: asyncio.subprocess.Process | None = None

    # ── Stop / reset API (thread-safe) ────────────────────────────────

    def stop(self) -> None:
        """Immediately stop TTS playback (thread-safe, for barge-in)."""
        self._stop_flag.set()
        proc = self._playback_process
        if proc is not None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    def reset_stop(self) -> None:
        """Clear the stop state before a new interaction."""
        self._stop_flag.clear()
        self._playback_process = None

    @property
    def stopped(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_flag.is_set()

    # ── Stoppable playback ────────────────────────────────────────────

    async def _play_with_tracking(self, audio_bytes: bytes) -> bool:
        """Play audio bytes while tracking the subprocess for interruption.

        Returns True if playback completed, False if stopped.
        """
        if not audio_bytes or self._stop_flag.is_set():
            return False

        player_name, cmd = _find_player()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        try:
            tmp.write(audio_bytes)
            tmp.close()

            process = await asyncio.create_subprocess_exec(
                *(cmd + [tmp.name]),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._playback_process = process

            wait_task = asyncio.create_task(process.wait())
            try:
                while not wait_task.done():
                    if self._stop_flag.is_set():
                        try:
                            process.terminate()
                        except (ProcessLookupError, OSError):
                            pass
                        try:
                            await asyncio.wait_for(process.wait(), timeout=0.5)
                        except (asyncio.TimeoutError, ProcessLookupError):
                            try:
                                process.kill()
                            except (ProcessLookupError, OSError):
                                pass
                        return False
                    await asyncio.sleep(0.05)
            finally:
                if not wait_task.done():
                    wait_task.cancel()
                    try:
                        await wait_task
                    except asyncio.CancelledError:
                        pass

            if process.returncode and process.returncode != 0:
                logger.warning(
                    "Audio player %s exited with code %d",
                    player_name, process.returncode,
                )
            return True
        except Exception:
            logger.warning("Stoppable playback error", exc_info=True)
            return False
        finally:
            self._playback_process = None
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    async def synthesize_and_play(
        self,
        text: str,
        tts_router,
        language: str = "id",
    ) -> float:
        """Split text into sentences, synthesize and play with overlap.

        Args:
            text: Full text to speak.
            tts_router: The TTS ProviderRouter instance.
            language: Language code for TTS voice selection.

        Returns:
            Total TTS time in seconds (synthesis of all sentences).
        """
        sentences = split_sentences(text)
        if not sentences:
            return 0.0

        # Single sentence — no need for the queue pipeline
        if len(sentences) == 1:
            return await self._synthesize_and_play_single(
                sentences[0], tts_router, language,
            )

        logger.info(
            "Streaming TTS: %d sentences from %d chars",
            len(sentences), len(text),
        )

        tts_start = time.perf_counter()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=0)
        first_audio_time: float | None = None

        async def producer() -> None:
            """Synthesize sentences and push audio to the queue."""
            nonlocal first_audio_time
            for i, sentence in enumerate(sentences):
                if self._stop_flag.is_set():
                    break
                try:
                    audio = await tts_router.execute(
                        "synthesize", _strip_markdown(sentence), language,
                    )
                    if i == 0 and first_audio_time is None:
                        first_audio_time = time.perf_counter() - tts_start
                    await audio_queue.put(audio)
                except Exception:
                    logger.warning(
                        "Streaming TTS: failed to synthesize sentence %d: %r",
                        i, sentence[:50], exc_info=True,
                    )
            await audio_queue.put(None)

        async def consumer() -> None:
            """Play audio chunks from the queue back-to-back."""
            while True:
                if self._stop_flag.is_set():
                    break
                audio = await audio_queue.get()
                if audio is None:
                    break
                completed = await self._play_with_tracking(audio)
                if not completed:
                    break

        await asyncio.gather(producer(), consumer())

        total_time = time.perf_counter() - tts_start
        logger.info(
            "Streaming TTS complete: %.2fs total, "
            "time-to-first-audio: %.2fs, %d sentences",
            total_time,
            first_audio_time or total_time,
            len(sentences),
        )
        return total_time

    async def _synthesize_and_play_single(
        self,
        text: str,
        tts_router,
        language: str,
    ) -> float:
        """Fast path for single-sentence responses (no queue overhead)."""
        if self._stop_flag.is_set():
            return 0.0
        start = time.perf_counter()
        try:
            audio = await tts_router.execute(
                "synthesize", _strip_markdown(text), language,
            )
            synth_time = time.perf_counter() - start
            logger.info(
                "TTS single sentence: %.2fs (%d bytes)",
                synth_time, len(audio),
            )
            await self._play_with_tracking(audio)
            return synth_time
        except Exception:
            logger.error(
                "Streaming TTS: single-sentence synthesis failed",
                exc_info=True,
            )
            return 0.0

    async def stream_from_llm(
        self,
        sentence_stream: AsyncIterator[str],
        tts_router,
        language: str = "auto",
    ) -> tuple[str, float]:
        """Stream sentences from LLM directly to TTS with overlapped playback.

        Unlike synthesize_and_play() which takes full text, this method
        accepts an async iterator of sentences from LLM streaming.
        Each sentence is synthesized and played as it arrives, so audio
        starts playing before the full LLM response is complete.

        Uses a sentence (text) queue with consumer-side prefetch:
        the producer pushes text quickly (never blocked by synthesis),
        and the consumer synthesizes with 1-item look-ahead so the next
        sentence's audio is being prepared while the current one plays.

        Args:
            sentence_stream: Async iterator yielding complete sentences
                from the LLM streaming response.
            tts_router: The TTS ProviderRouter instance.
            language: Language code ("id", "en", "auto").

        Returns:
            Tuple of (full response text, total time in seconds).
        """
        from nova.providers.tts.edge_tts_provider import detect_language

        tts_start = time.perf_counter()
        # Queue carries sentences (text), not audio — the consumer synthesizes.
        # Unbounded so the producer never blocks when the consumer stops early.
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=0)
        all_sentences: list[str] = []
        first_audio_time: float | None = None
        detected_lang = language

        async def producer() -> None:
            """Read sentences from LLM stream and push text to queue."""
            nonlocal detected_lang
            i = 0
            try:
                async for sentence in sentence_stream:
                    if self._stop_flag.is_set():
                        all_sentences.append(sentence)
                        break
                    all_sentences.append(sentence)

                    if detected_lang == "auto" and i == 0:
                        detected_lang = detect_language(sentence)
                        logger.debug(
                            "LLM→TTS stream: detected language=%s from %r",
                            detected_lang, sentence[:40],
                        )

                    await sentence_queue.put(sentence)
                    i += 1
            except asyncio.CancelledError:
                logger.debug("LLM→TTS producer cancelled (barge-in)")
            finally:
                try:
                    sentence_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        async def _synthesize(text: str) -> bytes | None:
            """Synthesize a single sentence, returning None on failure."""
            try:
                return await tts_router.execute(
                    "synthesize", _strip_markdown(text), detected_lang,
                )
            except Exception:
                logger.warning(
                    "LLM→TTS stream: synthesis failed: %r",
                    text[:50], exc_info=True,
                )
                return None

        async def consumer() -> None:
            """Synthesize and play with 1-item prefetch."""
            nonlocal first_audio_time
            prefetch_task: asyncio.Task | None = None
            done = False

            while not done:
                if self._stop_flag.is_set():
                    if prefetch_task:
                        prefetch_task.cancel()
                    break

                if prefetch_task is not None:
                    audio = await prefetch_task
                    prefetch_task = None
                else:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        break
                    if self._stop_flag.is_set():
                        break
                    audio = await _synthesize(sentence)
                    if audio is not None and first_audio_time is None:
                        first_audio_time = time.perf_counter() - tts_start

                if audio is None:
                    continue

                if not sentence_queue.empty():
                    try:
                        next_item = sentence_queue.get_nowait()
                        if next_item is None:
                            done = True
                        else:
                            prefetch_task = asyncio.create_task(
                                _synthesize(next_item),
                            )
                    except asyncio.QueueEmpty:
                        pass

                completed = await self._play_with_tracking(audio)
                if not completed:
                    if prefetch_task:
                        prefetch_task.cancel()
                    break

        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await consumer_task

        if not producer_task.done():
            producer_task.cancel()

        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            if not all_sentences:
                raise
            logger.warning(
                "LLM→TTS producer error (partial response available)",
                exc_info=True,
            )

        total_time = time.perf_counter() - tts_start
        full_text = " ".join(all_sentences)

        logger.info(
            "LLM→TTS stream complete: %.2fs total, "
            "time-to-first-audio: %.2fs, %d sentences, %d chars",
            total_time,
            first_audio_time or total_time,
            len(all_sentences),
            len(full_text),
        )

        return full_text, total_time


if __name__ == "__main__":
    # Quick test of sentence splitting
    test_texts = [
        "Halo! Saya Nova, asisten suara Anda. Saya bisa membantu banyak hal.",
        "Baterai Anda di 75%. Sedang mengisi daya.",
        "Dr. Budi mengatakan bahwa dll. itu penting. Benar sekali!",
        "Harganya Rp 3.500 per kg. Cukup murah.",
        "Ok.",
        "",
    ]
    for text in test_texts:
        result = split_sentences(text)
        print(f"Input:  {text!r}")
        print(f"Output: {result}")
        print()
