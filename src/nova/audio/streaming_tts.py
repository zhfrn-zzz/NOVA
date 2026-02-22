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
import re
import time
from collections.abc import AsyncIterator

from nova.audio.playback import play_audio

logger = logging.getLogger(__name__)

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
            if len(buffer) >= 10:
                sentences.append(buffer)
                buffer = part
            else:
                # Too short — merge with this part
                buffer = buffer + " " + part
        else:
            buffer = part

    # Flush remaining buffer
    if buffer:
        if sentences and len(buffer) < 10:
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
    """

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
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)
        first_audio_time: float | None = None

        async def producer() -> None:
            """Synthesize sentences and push audio to the queue."""
            nonlocal first_audio_time
            for i, sentence in enumerate(sentences):
                try:
                    audio = await tts_router.execute(
                        "synthesize", sentence, language,
                    )
                    if i == 0 and first_audio_time is None:
                        first_audio_time = time.perf_counter() - tts_start
                    await audio_queue.put(audio)
                except Exception:
                    logger.warning(
                        "Streaming TTS: failed to synthesize sentence %d: %r",
                        i, sentence[:50], exc_info=True,
                    )
                    # Skip this sentence, continue with next
            # Signal end of stream
            await audio_queue.put(None)

        async def consumer() -> None:
            """Play audio chunks from the queue back-to-back."""
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                try:
                    await play_audio(audio)
                except Exception:
                    logger.warning(
                        "Streaming TTS: playback error", exc_info=True,
                    )

        # Run producer and consumer concurrently.
        # Producer fills the queue while consumer drains it.
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
        start = time.perf_counter()
        try:
            audio = await tts_router.execute("synthesize", text, language)
            synth_time = time.perf_counter() - start
            logger.info(
                "TTS single sentence: %.2fs (%d bytes)",
                synth_time, len(audio),
            )
            await play_audio(audio)
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
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)
        all_sentences: list[str] = []
        first_audio_time: float | None = None
        detected_lang = language

        async def producer() -> None:
            nonlocal first_audio_time, detected_lang
            i = 0
            try:
                async for sentence in sentence_stream:
                    all_sentences.append(sentence)

                    # Auto-detect language from the first sentence
                    if detected_lang == "auto" and i == 0:
                        detected_lang = detect_language(sentence)
                        logger.debug(
                            "LLM→TTS stream: detected language=%s from %r",
                            detected_lang, sentence[:40],
                        )

                    try:
                        audio = await tts_router.execute(
                            "synthesize", sentence, detected_lang,
                        )
                        if i == 0 and first_audio_time is None:
                            first_audio_time = time.perf_counter() - tts_start
                        await audio_queue.put(audio)
                    except Exception:
                        logger.warning(
                            "LLM→TTS stream: synthesis failed for sentence %d: %r",
                            i, sentence[:50], exc_info=True,
                        )
                    i += 1
            finally:
                await audio_queue.put(None)

        async def consumer() -> None:
            while True:
                audio = await audio_queue.get()
                if audio is None:
                    break
                try:
                    await play_audio(audio)
                except Exception:
                    logger.warning(
                        "LLM→TTS stream: playback error", exc_info=True,
                    )

        await asyncio.gather(producer(), consumer())

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
