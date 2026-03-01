"""Custom sound loader — loads user audio files with generated fallback.

Checks the configured sounds directory for custom audio files.
If a custom file exists, loads and converts it. If not, falls back
to the generated sound function. Results are cached per sound name.

Supports wav/ogg/flac via soundfile, and mp3 via ffmpeg subprocess.
"""

import io
import logging
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path
from shutil import which

import numpy as np

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = (".wav", ".ogg", ".flac", ".mp3")

# In-memory cache: sound name → int16 numpy array
_cache: dict[str, np.ndarray] = {}


def _get_sounds_dir() -> Path:
    """Get the sounds directory from config, resolving relative paths."""
    from nova.config import get_config

    config = get_config()
    p = Path(config.custom_sounds_dir).expanduser()
    if not p.is_absolute():
        # Relative to project root (where pyproject.toml lives)
        p = Path(__file__).resolve().parents[3] / p
    return p


def load_sound(
    name: str,
    fallback_fn: Callable[..., np.ndarray],
    sample_rate: int = 22050,
    **fallback_kwargs,
) -> np.ndarray:
    """Load a custom sound file, or generate fallback.

    Checks {sounds_dir}/{name}.{wav,ogg,flac,mp3} for a user-provided file.
    If found, loads, converts to mono int16 at the target sample rate,
    and caches the result. If not found, calls fallback_fn.

    Args:
        name: Sound name without extension (e.g., "beep", "chime", "alert").
        fallback_fn: Function returning np.ndarray if no custom file found.
        sample_rate: Target sample rate.
        **fallback_kwargs: Extra kwargs passed to fallback_fn.

    Returns:
        Audio as int16 numpy array.
    """
    if name in _cache:
        return _cache[name]

    # Check config — skip file loading if disabled
    from nova.config import get_config

    config = get_config()
    if not config.custom_sounds_enabled:
        audio = fallback_fn(sample_rate=sample_rate, **fallback_kwargs)
        _cache[name] = audio
        return audio

    sounds_dir = _get_sounds_dir()
    sounds_dir.mkdir(parents=True, exist_ok=True)

    # Try each supported extension
    for ext in _SUPPORTED_EXTENSIONS:
        path = sounds_dir / f"{name}{ext}"
        if path.exists():
            try:
                audio = _load_audio_file(path, sample_rate)
                _cache[name] = audio
                logger.info("Loaded custom sound: %s", path)
                return audio
            except Exception:
                logger.warning(
                    "Failed to load %s, using fallback", path, exc_info=True,
                )

    # No custom file — use generated fallback
    audio = fallback_fn(sample_rate=sample_rate, **fallback_kwargs)
    _cache[name] = audio
    return audio


def _load_audio_file(path: Path, target_sr: int) -> np.ndarray:
    """Load an audio file and convert to mono int16 at target sample rate.

    Uses soundfile for wav/ogg/flac. For mp3, converts via ffmpeg first.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate in Hz.

    Returns:
        Audio as int16 numpy array.
    """
    if path.suffix.lower() == ".mp3":
        return _load_mp3_via_ffmpeg(path, target_sr)

    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")

    # Convert stereo (or multi-channel) to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed (linear interpolation)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    return (audio * 32767).astype(np.int16)


def _load_mp3_via_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    """Load an mp3 file by converting to WAV via ffmpeg.

    Args:
        path: Path to mp3 file.
        target_sr: Target sample rate in Hz.

    Returns:
        Audio as int16 numpy array.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        subprocess.CalledProcessError: If ffmpeg conversion fails.
    """
    ffmpeg = which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found — needed for mp3 support")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                ffmpeg, "-i", str(path),
                "-ar", str(target_sr),
                "-ac", "1",           # mono
                "-sample_fmt", "s16", # int16
                "-y", str(tmp_path),
            ],
            capture_output=True,
            timeout=15,
            check=True,
        )

        with wave.open(str(tmp_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).copy()
    finally:
        tmp_path.unlink(missing_ok=True)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 22050) -> bytes:
    """Convert a numpy int16 array to WAV bytes.

    Args:
        audio: Audio data as int16 numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        WAV-encoded bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def clear_cache() -> None:
    """Clear the sound cache (for testing)."""
    _cache.clear()
