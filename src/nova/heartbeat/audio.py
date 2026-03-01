"""Audio notification sounds for NOVA's heartbeat system.

Generates chime (gentle) and alert (urgent) notification sounds as
numpy int16 arrays. No external audio files needed — everything is
synthesized at runtime. Playback via sounddevice or temp wav fallback.
"""

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_chime(
    sample_rate: int = 22050,
    volume: float = 0.3,
) -> np.ndarray:
    """Generate a gentle two-note ascending chime (C5 → E5).

    Args:
        sample_rate: Audio sample rate in Hz.
        volume: Volume multiplier (0.0–1.0).

    Returns:
        Audio data as int16 numpy array.
    """
    duration = 0.6  # seconds
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float64)
    half = n_samples // 2

    # Two ascending notes: C5 (523 Hz) → E5 (659 Hz)
    note1 = np.sin(2 * np.pi * 523 * t[:half]) * volume
    note2 = np.sin(2 * np.pi * 659 * t[half:]) * volume

    # Smooth envelope (fade in → sustain → fade out)
    quarter = n_samples // 4
    envelope = np.concatenate([
        np.linspace(0, 1, quarter),
        np.ones(quarter),
        np.linspace(1, 0, n_samples - 2 * quarter),
    ])[:n_samples]

    audio = np.concatenate([note1, note2]) * envelope
    return (audio * 32767).astype(np.int16)


def generate_alert(
    sample_rate: int = 22050,
    volume: float = 0.5,
) -> np.ndarray:
    """Generate an urgent alternating two-tone alert (A5 / E5).

    Args:
        sample_rate: Audio sample rate in Hz.
        volume: Volume multiplier (0.0–1.0).

    Returns:
        Audio data as int16 numpy array.
    """
    duration = 1.0  # seconds
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float64)

    # Alternating tones: A5 (880 Hz) and E5 (660 Hz)
    freq = np.where(
        (t * 4).astype(int) % 2 == 0,
        880.0,  # A5
        660.0,  # E5
    )
    audio = np.sin(2 * np.pi * freq * t) * volume

    # Envelope: quick attack, long sustain, short release
    eighth = n_samples // 8
    envelope = np.concatenate([
        np.linspace(0, 1, eighth),
        np.ones(n_samples - 2 * eighth),
        np.linspace(1, 0, eighth),
    ])[:n_samples]

    audio = (audio * envelope * 32767).astype(np.int16)
    return audio


def play_notification_sound(
    audio: np.ndarray,
    sample_rate: int = 22050,
) -> None:
    """Play an audio array through the default output device.

    Tries sounddevice first, falls back to writing a temp wav file
    and playing via system command.

    Args:
        audio: int16 numpy audio array.
        sample_rate: Sample rate of the audio data.
    """
    try:
        import sounddevice as sd
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        return
    except ImportError:
        logger.debug("sounddevice not available, using wav fallback")
    except Exception as exc:
        logger.warning("sounddevice playback failed: %s", exc)

    # Fallback: write temp wav and play with system command
    _play_via_temp_wav(audio, sample_rate)


def _play_via_temp_wav(audio: np.ndarray, sample_rate: int) -> None:
    """Write audio to a temp wav file and play via system command."""
    import platform
    import subprocess

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16 = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(audio.tobytes())

        system = platform.system()
        if system == "Linux":
            subprocess.run(
                ["aplay", str(tmp_path)],
                capture_output=True, timeout=10,
            )
        elif system == "Darwin":
            subprocess.run(
                ["afplay", str(tmp_path)],
                capture_output=True, timeout=10,
            )
        elif system == "Windows":
            # Use PowerShell to play wav
            subprocess.run(
                [
                    "powershell", "-c",
                    f"(New-Object Media.SoundPlayer '{tmp_path}').PlaySync()",
                ],
                capture_output=True, timeout=10,
            )
        else:
            logger.warning("Unsupported platform for wav playback: %s", system)
    except Exception as exc:
        logger.warning("Wav fallback playback failed: %s", exc)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
