"""NOVA structured logging — Rich console + daily-rotated file logs."""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Shared console instance for rich output
console = Console()

# Log directory: ~/.nova/logs/
_LOG_DIR = Path.home() / ".nova" / "logs"

# File log format: [TIME] [LEVEL] [COMPONENT] message
_FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)-20s] %(message)s"
_FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """Configure NOVA logging with Rich console output and daily file rotation.

    Args:
        verbose: If True, sets log level to DEBUG.
        log_level: Default log level string (e.g. "INFO", "WARNING").
    """
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)

    # Create log directory
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Root logger ---
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates on reconfigure
    root.handlers.clear()

    # --- Console handler (Rich) ---
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=verbose,
        show_level=True,
        show_time=True,
    )
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)

    # --- File handler (daily rotation, keep 7 days) ---
    log_file = _LOG_DIR / "nova.log"
    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FMT))
    root.addHandler(file_handler)

    # --- Quiet noisy third-party loggers ---
    for name in ("httpx", "httpcore", "aiohttp", "hpack", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging configured: level=%s, file=%s, verbose=%s",
        logging.getLevelName(level), log_file, verbose,
    )


def get_log_dir() -> Path:
    """Return the log directory path."""
    return _LOG_DIR
