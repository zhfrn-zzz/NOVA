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
        "--push-to-talk",
        action="store_true",
        help="Push-to-talk mode (press Enter to speak, as in Phase 1)",
    )
    parser.add_argument(
        "--hotkey",
        action="store_true",
        help="Use keyboard hotkey (Ctrl+Space) instead of wake word detection",
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
    parser.add_argument(
        "--quota",
        action="store_true",
        help="Show Google Cloud TTS quota usage for the current month",
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

    # Check wake word model
    config = get_config()
    try:
        import os

        from nova.audio.wake_word_oww import OpenWakeWordDetector  # noqa: F401

        model_exists = os.path.isfile(config.wake_word_model_path)
        if model_exists:
            console.print(
                f"  [green]✅[/] wake_word: {config.wake_word_model_path} found"
            )
        else:
            console.print(
                f"  [red]❌[/] wake_word: {config.wake_word_model_path} not found"
            )
            all_ok = False
    except ImportError:
        console.print("  [yellow]⚠️[/] wake_word: openwakeword not installed (hotkey fallback)")

    # Check Google Cloud TTS quota status
    try:
        from nova.providers.tts.google_cloud_tts import GoogleCloudTTSProvider

        gcp_tts = GoogleCloudTTSProvider()
        status = gcp_tts.get_quota_status()
        if status["configured"]:
            remaining = status["remaining"]
            if remaining > 1000:
                console.print(
                    f"  [green]✅[/] google_cloud_tts: connected "
                    f"({status['chars_used']:,} / {status['limit']:,} chars used)"
                )
            else:
                console.print(
                    f"  [yellow]⚠️[/] google_cloud_tts: quota exceeded "
                    f"({status['chars_used']:,} / {status['limit']:,} chars used)"
                )
                all_ok = False
        else:
            console.print("  [dim]ℹ️[/]  google_cloud_tts: not configured")
    except Exception:
        console.print("  [dim]ℹ️[/]  google_cloud_tts: not configured")

    console.print()
    if all_ok:
        console.print("[bold green]All systems operational.[/]\n")
    else:
        console.print(
            "[bold yellow]Some components unavailable"
            " — NOVA may have reduced functionality.[/]\n"
        )


def _run_quota() -> None:
    """Display Google Cloud TTS quota usage for the current month."""
    console.print("\n[bold]Google Cloud TTS Quota[/]\n")

    try:
        from nova.providers.tts.google_cloud_tts import GoogleCloudTTSProvider

        provider = GoogleCloudTTSProvider()
        status = provider.get_quota_status()

        if not status["configured"]:
            console.print(
                "  [dim]Google Cloud TTS is not configured.[/]\n"
                "  Set NOVA_GOOGLE_CLOUD_TTS_KEY_PATH in .env to enable.\n"
            )
            return

        # Parse month string (e.g. "2026-02") to human-readable
        month_str = status["month"]
        try:
            from datetime import datetime
            month_dt = datetime.strptime(month_str, "%Y-%m")
            month_display = month_dt.strftime("%B %Y")
        except (ValueError, TypeError):
            month_display = month_str

        chars_used = status["chars_used"]
        limit = status["limit"]
        remaining = status["remaining"]
        pct = (chars_used / limit * 100) if limit > 0 else 0

        if remaining > 1000:
            color = "green"
        elif remaining > 0:
            color = "yellow"
        else:
            color = "red"

        console.print(
            f"  [{color}]Google Cloud TTS: {chars_used:,} / {limit:,} "
            f"characters used ({month_display})[/]"
        )
        console.print(f"  Remaining: {remaining:,} characters ({100 - pct:.1f}%)\n")

    except Exception as e:
        console.print(f"  [red]Error reading quota: {e}[/]\n")


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
        "[bold green]NOVA[/] ready (push-to-talk voice mode). "
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


async def _wake_word_mode(orchestrator, force_hotkey: bool = False) -> None:
    """Run the wake-word continuous listening mode.

    By default uses OpenWakeWord for always-listening detection.
    Falls back to hotkey mode if openwakeword fails to load or
    if force_hotkey is True.
    """
    config = get_config()
    loop = asyncio.get_event_loop()
    detector = None
    mode_label = "wake word"

    if not force_hotkey:
        try:
            from nova.audio.wake_word_oww import OpenWakeWordDetector

            detector = OpenWakeWordDetector()
            detector.start(loop)
            mode_label = f"wake word ({config.wake_word_model_path})"
            console.print(
                f"[bold green]NOVA[/] ready ({mode_label}). "
                f"Say the wake word to activate, or type 'exit' to quit.\n"
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "OpenWakeWord failed to load (%s), falling back to hotkey", e,
            )
            detector = None

    if detector is None:
        from nova.audio.wake_word import HotkeyWakeWordDetector

        detector = HotkeyWakeWordDetector()
        detector.start(loop)
        mode_label = f"hotkey ({config.wake_word_hotkey})"
        console.print(
            f"[bold green]NOVA[/] ready ({mode_label}). "
            f"Press [bold]{config.wake_word_hotkey}[/] to activate, "
            f"or type 'exit' to quit.\n"
        )

    text_fallback = False

    # Run a background task for keyboard exit input
    exit_event = asyncio.Event()

    async def _exit_listener():
        """Listen for typed 'exit' commands in background."""
        while not exit_event.is_set():
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input(),
                )
                stripped = user_input.strip().lower()
                if stripped in ("exit", "quit", "bye"):
                    exit_event.set()
                    return
                # If they typed actual text, process it
                if user_input.strip():
                    try:
                        response = await orchestrator.handle_interaction(user_input.strip())
                        console.print(f"[bold cyan]Nova:[/] {response}\n")
                    except Exception:
                        logging.getLogger(__name__).exception("Error")
                        console.print("[red]Terjadi kesalahan.[/]\n")
            except EOFError:
                exit_event.set()
                return

    exit_task = asyncio.create_task(_exit_listener())

    try:
        while not exit_event.is_set():
            # Wait for either hotkey activation or exit
            activation_task = asyncio.create_task(detector.wait_for_activation())

            done, pending = await asyncio.wait(
                [activation_task, exit_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                if task is activation_task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            if exit_event.is_set():
                break

            if activation_task in done:
                # Hotkey was pressed — capture and process voice
                console.print("[bold yellow]🎤 Listening...[/]")

                try:
                    response = await orchestrator.handle_voice_interaction()

                    if response == "__AUDIO_DEVICE_ERROR__":
                        console.print(
                            "[red]Mikrofon tidak ditemukan, beralih ke mode teks.[/]\n"
                        )
                        text_fallback = True
                        break

                    if response == "__STT_FAILED__":
                        console.print(
                            "[yellow]Maaf, saya tidak bisa mendengar sekarang.[/]\n"
                        )
                        continue

                    if response is None:
                        console.print(
                            "[dim]Saya tidak mendengar apa-apa, bisa diulang?[/]\n"
                        )
                        continue

                    transcript = orchestrator.last_transcript
                    if transcript:
                        console.print(f"[bold white]You:[/] {transcript}")
                    console.print(f"[bold cyan]Nova:[/] {response}\n")

                except Exception:
                    logging.getLogger(__name__).exception("Voice interaction error")
                    console.print("[red]Terjadi kesalahan.[/]\n")

    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        exit_task.cancel()
        try:
            await exit_task
        except asyncio.CancelledError:
            pass

    # If mic failed, fall back to text mode
    if text_fallback:
        await _text_mode(orchestrator)


async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()

    config = get_config()
    setup_logging(verbose=args.verbose, log_level=config.log_level)

    # --check mode: test all providers and exit
    if args.check:
        await _run_check()
        return

    # --quota mode: show Google TTS quota and exit
    if args.quota:
        _run_quota()
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
    elif args.push_to_talk:
        await _voice_mode(orchestrator)
    elif args.hotkey:
        # Forced hotkey mode
        await _wake_word_mode(orchestrator, force_hotkey=True)
    else:
        # Default: OpenWakeWord always-listening (hotkey fallback)
        await _wake_word_mode(orchestrator)

    console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")


if __name__ == "__main__":
    main()
