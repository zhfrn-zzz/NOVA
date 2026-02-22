"""NOVA entry point — CLI args, async loop, and user interaction."""

import argparse
import asyncio
import logging
import sys

from rich.console import Console

from nova.config import get_config
from nova.utils.logger import setup_logging

console = Console()


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connectivity to all providers, mic, and audio player",
    )
    return parser.parse_args()


async def _run_check() -> None:
    """Test connectivity to all providers, microphone, and audio player."""
    from nova.orchestrator import Orchestrator

    console.print("\n[bold]NOVA System Check[/]\n")

    orchestrator = Orchestrator()
    results = await orchestrator.check_providers()

    all_ok = True
    for component, info in results.items():
        available = info["available"]
        status = info["status"]
        if available:
            console.print(f"  [green]✅[/] {component}: {status}")
        else:
            console.print(f"  [red]❌[/] {component}: {status}")
            all_ok = False

    console.print()
    if all_ok:
        console.print("[bold green]All systems operational.[/]\n")
    else:
        console.print(
            "[bold yellow]Some components unavailable"
            " — NOVA may have reduced functionality.[/]\n"
        )


async def _text_mode(orchestrator) -> None:
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
            console.print("[red]Terjadi kesalahan, tapi saya masih berjalan.[/]\n")


async def _voice_mode(orchestrator) -> None:
    """Run the push-to-talk voice interactive loop."""
    console.print(
        "[bold green]NOVA[/] ready (voice mode). "
        "Press [bold]Enter[/] to speak, type 'exit' to quit.\n"
    )

    loop = asyncio.get_event_loop()
    text_fallback = False  # Set to True if mic fails

    while True:
        try:
            if text_fallback:
                prompt = "Type your message (or 'exit'): "
            else:
                prompt = "Press Enter to speak (or type 'exit'): "

            user_input = await loop.run_in_executor(
                None, lambda: input(prompt),
            )
        except EOFError:
            break

        # Allow typing exit/quit/bye to leave
        stripped = user_input.strip().lower()
        if stripped in ("exit", "quit", "bye"):
            break

        # If they typed actual text, use text mode for it
        if user_input.strip():
            try:
                response = await orchestrator.handle_interaction(user_input.strip())
                console.print(f"[bold cyan]Nova:[/] {response}\n")
            except KeyboardInterrupt:
                break
            except Exception:
                logging.getLogger(__name__).exception("Error during interaction")
                console.print("[red]Terjadi kesalahan, tapi saya masih berjalan.[/]\n")
            continue

        # Text fallback mode — don't try to record
        if text_fallback:
            continue

        # Push-to-talk: Enter was pressed with no text
        console.print("[bold yellow]🎤 Listening...[/]")

        try:
            response = await orchestrator.handle_voice_interaction()

            # Handle sentinel values from orchestrator
            if response == "__AUDIO_DEVICE_ERROR__":
                console.print(
                    "[red]Mikrofon tidak ditemukan, beralih ke mode teks.[/]\n"
                )
                text_fallback = True
                continue

            if response == "__STT_FAILED__":
                console.print(
                    "[yellow]Maaf, saya tidak bisa mendengar sekarang. "
                    "Coba ketik saja.[/]\n"
                )
                text_fallback = True
                continue

            if response is None:
                console.print("[dim]Saya tidak mendengar apa-apa, bisa diulang?[/]\n")
                continue

            # Print what was heard and the response
            transcript = orchestrator.last_transcript
            if transcript:
                console.print(f"[bold white]You:[/] {transcript}")
            console.print(f"[bold cyan]Nova:[/] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/]\n")
        except Exception:
            logging.getLogger(__name__).exception("Error during voice interaction")
            console.print("[red]Terjadi kesalahan, tapi saya masih berjalan.[/]\n")


async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()

    config = get_config()
    setup_logging(verbose=args.verbose, log_level=config.log_level)

    # --check mode: test all providers and exit
    if args.check:
        await _run_check()
        return

    try:
        config.validate_api_keys()
    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        sys.exit(1)

    from nova.orchestrator import Orchestrator

    orchestrator = Orchestrator()

    if args.text_only:
        await _text_mode(orchestrator)
    else:
        await _voice_mode(orchestrator)

    console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


if __name__ == "__main__":
    main()
