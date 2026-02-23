"""Double-clap detector for alternative NOVA activation.

Analyzes raw int16 audio frames (shared with the OpenWakeWord reader loop)
to detect two sharp transient spikes separated by a configurable gap.  No
extra threads, no extra audio streams — pure frame-by-frame computation.
"""

import collections
import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)

# Frame size matches OpenWakeWord: 80ms at 16kHz = 1280 samples
_FRAME_SAMPLES = 1280
_SAMPLE_RATE = 16000
_FRAME_DURATION_S = _FRAME_SAMPLES / _SAMPLE_RATE  # 0.08s

# Rolling ambient window: ~2 seconds of frames
_AMBIENT_WINDOW = int(2.0 / _FRAME_DURATION_S)  # 25 frames

# Cooldown after trigger: ~2 seconds
_COOLDOWN_FRAMES = int(2.0 / _FRAME_DURATION_S)  # 25 frames


class ClapDetector:
    """Detects double-clap patterns in audio frames.

    Feed each 80ms int16 audio frame via ``process_frame``.  Returns
    ``True`` when a valid double-clap pattern is detected.

    Detection logic:
        1. Calculate frame RMS energy.
        2. Maintain rolling ambient RMS over ~2 seconds.
        3. Spike = RMS > ambient * energy_multiplier (single frame ≤80ms).
        4. After first spike, wait for second spike within min–max gap.
        5. Two spikes in window → trigger.
        6. 2-second cooldown after trigger.
    """

    def __init__(
        self,
        energy_multiplier: float = 15.0,
        min_rms: float = 500.0,
        min_gap_ms: int = 200,
        max_gap_ms: int = 600,
    ) -> None:
        self._energy_multiplier = energy_multiplier
        self._min_rms = min_rms
        self._min_gap_s = min_gap_ms / 1000.0
        self._max_gap_s = max_gap_ms / 1000.0

        # Rolling ambient RMS tracker
        self._ambient_history: collections.deque[float] = collections.deque(
            maxlen=_AMBIENT_WINDOW,
        )
        self._ambient_rms: float = 0.0

        # State machine
        self._first_clap_time: float | None = None
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Analyze one audio frame for double-clap detection.

        Args:
            audio_frame: 1-D int16 numpy array (1280 samples, 80ms at 16kHz).

        Returns:
            True if a double-clap was detected.
        """
        # Cooldown — skip processing
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        rms = self._compute_rms(audio_frame)

        # Update ambient noise level
        is_spike = self._is_spike(rms)

        # Only include non-spike frames in ambient baseline
        if not is_spike:
            self._ambient_history.append(rms)
            self._ambient_rms = self._rolling_mean()

        # State machine: waiting for first or second clap
        now = time.monotonic()

        if self._first_clap_time is not None:
            elapsed = now - self._first_clap_time

            if elapsed > self._max_gap_s:
                # Window expired — reset
                self._first_clap_time = None

            elif is_spike and elapsed >= self._min_gap_s:
                # Second clap in window → trigger!
                self._first_clap_time = None
                self._cooldown_remaining = _COOLDOWN_FRAMES
                logger.info(
                    "Double clap detected! (gap=%.0fms, rms=%.1f, ambient=%.1f)",
                    elapsed * 1000,
                    rms,
                    self._ambient_rms,
                )
                return True

        elif is_spike:
            # First clap
            self._first_clap_time = now
            logger.debug(
                "First clap candidate (rms=%.1f, ambient=%.1f, threshold=%.1f)",
                rms,
                self._ambient_rms,
                self._ambient_rms * self._energy_multiplier,
            )

        return False

    def reset(self) -> None:
        """Reset internal state (e.g. after activation)."""
        self._first_clap_time = None
        self._cooldown_remaining = _COOLDOWN_FRAMES

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rms(frame: np.ndarray) -> float:
        """Compute RMS energy of an int16 audio frame."""
        samples = frame.astype(np.float64)
        return float(math.sqrt(np.mean(samples * samples)))

    def _is_spike(self, rms: float) -> bool:
        """Check if *rms* qualifies as a clap spike above ambient.

        Both conditions must be true:
            1. RMS > ambient * multiplier (relative to background)
            2. RMS > min_rms (absolute floor — filters keyboard, etc.)
        """
        # Absolute floor — keyboard typing (~36 RMS) never passes
        if rms < self._min_rms:
            return False
        if self._ambient_rms < 1.0:
            # Not enough ambient data yet — absolute check is enough
            return True
        return rms > self._ambient_rms * self._energy_multiplier

    def _rolling_mean(self) -> float:
        """Mean of the ambient history deque."""
        if not self._ambient_history:
            return 0.0
        return sum(self._ambient_history) / len(self._ambient_history)
