"""NOVA entry point — CLI args, async loop, and user interaction."""

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

from nova.config import get_config
from nova.orchestrator import Orchestrator

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    config = get_config()
    if not verbose:
        level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="nova",
        description="NOVA — Neural-Orchestrated Voice Assistant",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text input mode (no microphone)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def _text_mode(orchestrator: Orchestrator) -> None:
    """Run the text-only interactive loop."""
    console.print("[bold green]NOVA[/] ready (text mode). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("You: "),
            )
        except EOFError:
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            break

        try:
            response = await orchestrator.handle_interaction(user_input)
            console.print(f"[bold cyan]Nova:[/] {response}\n")
        except KeyboardInterrupt:
            break
        except Exception:
            logging.getLogger(__name__).exception("Error during interaction")
            console.print("[red]An error occurred. Please try again.[/]\n")


async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()
    _setup_logging(verbose=args.verbose)

    config = get_config()
    try:
        config.validate_api_keys()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        sys.exit(1)

    orchestrator = Orchestrator()

    if args.text_only:
        await _text_mode(orchestrator)
    else:
        # Default to text mode for now; voice mode added in Task 9
        await _text_mode(orchestrator)

    console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


if __name__ == "__main__":
    main()
