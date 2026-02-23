"""TTS quota tracker — tracks monthly Google Cloud TTS character usage.

Stores usage in ~/.nova/tts_usage.json with atomic writes for thread safety.
Prevents exceeding the free-tier quota by refusing requests that would
push usage over the configured limit.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_USAGE_DIR = Path.home() / ".nova"
_USAGE_FILE = _USAGE_DIR / "tts_usage.json"


class TTSQuotaTracker:
    """Tracks Google Cloud TTS monthly character usage with file-based persistence.

    Thread-safe via atomic writes (write to temp file, then os.replace).
    """

    def __init__(self, monthly_limit: int = 950_000) -> None:
        self._monthly_limit = monthly_limit
        self._usage_file = _USAGE_FILE
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create usage file with defaults if it doesn't exist."""
        if not self._usage_file.exists():
            _USAGE_DIR.mkdir(parents=True, exist_ok=True)
            self._write_data({
                "month": datetime.now(tz=timezone.utc).strftime("%Y-%m"),
                "chars_used": 0,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            })
            logger.debug("Created TTS usage file: %s", self._usage_file)

    def _read_data(self) -> dict:
        """Read usage data from disk."""
        try:
            with open(self._usage_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt TTS usage file, resetting")
            return {
                "month": datetime.now(tz=timezone.utc).strftime("%Y-%m"),
                "chars_used": 0,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            }

    def _write_data(self, data: dict) -> None:
        """Write usage data atomically (temp file + os.replace)."""
        _USAGE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(_USAGE_DIR), suffix=".tmp", prefix="tts_usage_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, str(self._usage_file))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def reset_if_new_month(self) -> None:
        """Reset character count if the current month differs from stored month."""
        data = self._read_data()
        current_month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
        if data.get("month") != current_month:
            logger.info(
                "New month detected (%s → %s), resetting TTS quota",
                data.get("month"), current_month,
            )
            self._write_data({
                "month": current_month,
                "chars_used": 0,
                "last_updated": datetime.now(tz=timezone.utc).isoformat(),
            })

    def can_use(self, text_length: int) -> bool:
        """Check if using text_length chars would stay under the quota.

        Args:
            text_length: Number of characters to check.

        Returns:
            True if chars_used + text_length < monthly_limit.
        """
        data = self._read_data()
        return data.get("chars_used", 0) + text_length < self._monthly_limit

    def record_usage(self, text_length: int) -> None:
        """Record text_length characters of usage.

        Args:
            text_length: Number of characters used.
        """
        data = self._read_data()
        data["chars_used"] = data.get("chars_used", 0) + text_length
        data["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
        self._write_data(data)
        logger.debug(
            "TTS quota: %d / %d chars used",
            data["chars_used"], self._monthly_limit,
        )

    def get_remaining(self) -> int:
        """Return remaining characters available this month.

        Returns:
            Number of characters remaining (monthly_limit - chars_used).
        """
        data = self._read_data()
        return self._monthly_limit - data.get("chars_used", 0)

    def get_usage(self) -> tuple[int, int, str]:
        """Return current usage stats.

        Returns:
            Tuple of (chars_used, monthly_limit, month_string).
        """
        data = self._read_data()
        return (
            data.get("chars_used", 0),
            self._monthly_limit,
            data.get("month", "unknown"),
        )
