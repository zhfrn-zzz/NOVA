"""NOVA entry point — CLI args, async loop, and user interaction."""

import argparse
import asyncio
import logging
import sys

from rich.console import Console

from nova.config import get_config
from nova.deeptalk.session import is_deeptalk_trigger
from nova.utils.logger import setup_logging

console = Console()

_barge_in_detector = None  # Singleton, lazy-initialized


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


# ── Heartbeat notification helpers ─────────────────────────────────────


async def _check_text_notifications(orchestrator) -> None:
    """Check for ACTIVE heartbeat notifications in text/push-to-talk mode.

    Only handles ACTIVE notifications (printed to console).
    PASSIVE and GENTLE notifications are handled by the orchestrator
    (injected into LLM context on next interaction).
    """
    from nova.heartbeat.queue import Urgency

    queue = orchestrator.notification_queue
    if not queue.has_urgent():
        return

    # Peek: only handle ACTIVE notifications here.
    # GENTLE notifications are left in the queue for the orchestrator
    # to inject into LLM context (via get_passive_and_gentle in text mode).
    notif = queue.get_next_urgent()
    if notif is None:
        return

    if notif.urgency != Urgency.ACTIVE:
        # Not ACTIVE — put it back for orchestrator to handle via context
        notif.urgency = Urgency.PASSIVE
        queue.push(notif)
        return

    # Format ACTIVE notification for console display
    if notif.message == "__morning_greeting__":
        msg = "Selamat pagi, Pak."
    elif notif.message == "__sleep_reminder__":
        msg = "Pak, sudah malam. Sebaiknya istirahat."
    else:
        msg = notif.message

    console.print(f"\n[bold yellow]🔔 [NOVA alert][/] {msg}\n")


async def _check_voice_notifications(orchestrator, detector, config) -> bool:
    """Check and handle urgent notifications with audio in wake word mode.

    Returns True if a notification was handled (caller should re-check queue).
    """
    queue = orchestrator.notification_queue
    if not queue.has_urgent():
        return False

    from nova.heartbeat.audio import get_alert, get_chime, play_notification_sound
    from nova.heartbeat.queue import Urgency

    notif = queue.get_next_urgent()
    if notif is None:
        return False

    logger = logging.getLogger(__name__)

    if notif.urgency == Urgency.GENTLE:
        # 1. Pause wake word detector
        detector.stop()

        # 2. Play chime
        try:
            chime = get_chime(volume=config.chime_volume)
            play_notification_sound(chime)
        except Exception:
            logger.warning("Chime playback failed", exc_info=True)

        # 3. Listen for user response (short timeout)
        try:
            from nova.audio.capture import AudioCapture
            capture = AudioCapture()
            capture.max_recording_seconds = config.gentle_listen_timeout
            wav_bytes = await capture.capture()

            if len(wav_bytes) > 44:
                # User responded — process as normal with notification in context

                # Inject notification as passive so it's in the LLM context
                notif.urgency = Urgency.PASSIVE
                queue.push(notif)

                # Transcribe and handle
                transcript = await orchestrator._stt_router.execute(
                    "transcribe", wav_bytes,
                )
                if transcript and transcript.strip():
                    response = await orchestrator.handle_interaction(
                        transcript.strip()
                    )
                    console.print(f"[bold white]You:[/] {transcript.strip()}")
                    console.print(f"[bold cyan]Nova:[/] {response}\n")
            else:
                # No response — retry or downgrade
                notif.attempts += 1
                if notif.attempts < notif.max_attempts:
                    queue.push(notif)  # re-queue for later
                else:
                    notif.urgency = Urgency.PASSIVE
                    queue.push(notif)  # downgrade to passive
        except Exception:
            logger.warning("Gentle notification listen failed", exc_info=True)
            # Re-queue as passive on failure
            notif.urgency = Urgency.PASSIVE
            queue.push(notif)

        # 4. Resume wake word detector
        import asyncio
        loop = asyncio.get_event_loop()
        detector.start(loop)
        return True

    elif notif.urgency == Urgency.ACTIVE:
        # 1. Pause wake word detector
        detector.stop()

        # 2. Play alert sound (loop 3x before speaking)
        try:
            alert = get_alert(volume=config.alert_volume)
            play_notification_sound(alert, repeat=3)
        except Exception:
            logger.warning("Alert playback failed", exc_info=True)

        # 3. Generate and speak notification via LLM + TTS
        try:
            await orchestrator.deliver_notification(notif)
        except Exception:
            logger.exception("Active notification delivery failed")
            # Fallback: print to console
            console.print(
                f"\n[bold red]🔔 [NOVA alert][/] {notif.message}\n"
            )

        # 4. Resume wake word detector
        import asyncio
        loop = asyncio.get_event_loop()
        detector.start(loop)
        return True

    return False


def _get_barge_in_detector():
    """Lazy-initialize the barge-in detector (loads Silero VAD ONNX)."""
    global _barge_in_detector
    if _barge_in_detector is None:
        from nova.deeptalk.barge_in import BargeInDetector

        _barge_in_detector = BargeInDetector()
    return _barge_in_detector


async def _run_deeptalk(orchestrator) -> None:
    """Run a DeepTalk continuous conversation session.

    Blocks until the user says an exit phrase or the session errors out.
    """
    from nova.deeptalk.session import DeepTalkSession

    try:
        detector = _get_barge_in_detector()
    except Exception:
        logging.getLogger(__name__).exception("Failed to initialize barge-in detector")
        console.print("[red]Gagal memuat model VAD untuk DeepTalk.[/]\n")
        return

    session = DeepTalkSession(orchestrator, detector)
    await session.start()


async def _text_mode(orchestrator) -> None:
    """Run the text-only interactive loop."""
    orchestrator._text_only = True
    console.print("[bold green]NOVA[/] ready (text mode). Type 'exit' to quit.\n")

    try:
        while True:
            # Check for urgent heartbeat notifications (text mode: print to console)
            await _check_text_notifications(orchestrator)

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
    finally:
        orchestrator.stop()


async def _voice_mode(orchestrator) -> None:
    """Run the push-to-talk voice interactive loop."""
    console.print(
        "[bold green]NOVA[/] ready (push-to-talk voice mode). "
        "Press [bold]Enter[/] to speak, type 'exit' to quit.\n"
    )

    loop = asyncio.get_event_loop()
    text_fallback = False  # Set to True if mic fails

    try:
        while True:
            # Check for urgent heartbeat notifications
            await _check_text_notifications(orchestrator)

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

            # DeepTalk keyboard entry
            if stripped in ("d", "deeptalk"):
                await _run_deeptalk(orchestrator)
                continue

            # If they typed actual text, use text mode for it
            if user_input.strip():
                try:
                    response = await orchestrator.handle_interaction(user_input.strip())
                    console.print(f"[bold cyan]Nova:[/] {response}\n")
                except KeyboardInterrupt:
                    break
                except Exception:
                    logging.getLogger(__name__).exception("Error during interaction")
                    console.print(
                        "[red]Terjadi kesalahan, tapi saya masih berjalan.[/]\n"
                    )
                continue

            # Text fallback mode — don't try to record
            if text_fallback:
                continue

            # Push-to-talk: Enter was pressed with no text
            console.print("[bold yellow]🎤 Listening...[/]")

            try:
                transcript = await orchestrator.capture_and_transcribe()

                if transcript == "__AUDIO_DEVICE_ERROR__":
                    console.print(
                        "[red]Mikrofon tidak ditemukan, beralih ke mode teks.[/]\n"
                    )
                    text_fallback = True
                    continue

                if transcript == "__STT_FAILED__":
                    console.print(
                        "[yellow]Maaf, saya tidak bisa mendengar sekarang. "
                        "Coba ketik saja.[/]\n"
                    )
                    text_fallback = True
                    continue

                if not transcript:
                    console.print(
                        "[dim]Saya tidak mendengar apa-apa, bisa diulang?[/]\n"
                    )
                    continue

                # Check for DeepTalk voice trigger
                if is_deeptalk_trigger(transcript):
                    await _run_deeptalk(orchestrator)
                    continue

                response = await orchestrator.handle_interaction(transcript)
                console.print(f"[bold white]You:[/] {transcript}")
                console.print(f"[bold cyan]Nova:[/] {response}\n")

            except KeyboardInterrupt:
                console.print("\n[dim]Cancelled.[/]\n")
            except Exception:
                logging.getLogger(__name__).exception(
                    "Error during voice interaction"
                )
                console.print(
                    "[red]Terjadi kesalahan, tapi saya masih berjalan.[/]\n"
                )
    finally:
        orchestrator.stop()


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

            # Wire ambient RMS for heartbeat presence heuristic
            if hasattr(detector, "get_ambient_rms"):
                orchestrator.set_ambient_fn(detector.get_ambient_rms)
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
    deeptalk_event = asyncio.Event()

    async def _exit_listener():
        """Listen for typed 'exit' / 'deeptalk' commands in background."""
        while not exit_event.is_set():
            try:
                user_input = await loop.run_in_executor(
                    None, lambda: input(),
                )
                stripped = user_input.strip().lower()
                if stripped in ("exit", "quit", "bye"):
                    exit_event.set()
                    return
                if stripped in ("d", "deeptalk"):
                    deeptalk_event.set()
                    continue
                # If they typed actual text, process it
                if user_input.strip():
                    try:
                        response = await orchestrator.handle_interaction(
                            user_input.strip()
                        )
                        console.print(f"[bold cyan]Nova:[/] {response}\n")
                    except Exception:
                        logging.getLogger(__name__).exception("Error")
                        console.print("[red]Terjadi kesalahan.[/]\n")
            except (EOFError, OSError):
                exit_event.set()
                return
            except Exception:
                logging.getLogger(__name__).debug(
                    "exit_listener error", exc_info=True,
                )
                await asyncio.sleep(0.5)

    exit_task = asyncio.create_task(_exit_listener())

    activation_task = None
    try:
        while not exit_event.is_set():
            # --- DeepTalk keyboard trigger ---
            if deeptalk_event.is_set():
                deeptalk_event.clear()
                detector.stop()
                try:
                    await _run_deeptalk(orchestrator)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "DeepTalk session error (keyboard)"
                    )
                detector.start(loop)
                activation_task = None
                continue

            # --- Check heartbeat notification queue ---
            handled = await _check_voice_notifications(
                orchestrator, detector, config,
            )
            if handled:
                activation_task = None  # reset after notification handling
                continue  # Re-check queue before waiting for wake word

            # Reuse existing activation task if still running
            if activation_task is None or activation_task.done():
                activation_task = asyncio.create_task(detector.wait_for_activation())

            done, pending = await asyncio.wait(
                [activation_task, exit_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5,
            )

            if not done:
                continue

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
                console.print("[bold yellow]🎤 Listening...[/]")
                _logger = logging.getLogger(__name__)

                try:
                    transcript = await orchestrator.capture_and_transcribe()
                    _logger.debug("Wake word transcript: %r", transcript)

                    if transcript == "__AUDIO_DEVICE_ERROR__":
                        console.print(
                            "[red]Mikrofon tidak ditemukan, "
                            "beralih ke mode teks.[/]\n"
                        )
                        text_fallback = True
                        break

                    if transcript == "__STT_FAILED__":
                        console.print(
                            "[yellow]Maaf, saya tidak bisa "
                            "mendengar sekarang.[/]\n"
                        )
                        continue

                    if not transcript:
                        console.print(
                            "[dim]Saya tidak mendengar apa-apa, "
                            "bisa diulang?[/]\n"
                        )
                        continue

                    # Check for DeepTalk voice trigger
                    if is_deeptalk_trigger(transcript):
                        _logger.info(
                            "DeepTalk trigger: %r", transcript,
                        )
                        sys.stderr.flush()
                        _logger.info("Stopping wake word detector...")
                        sys.stderr.flush()
                        detector.stop()
                        _logger.info("Detector stopped, entering DeepTalk")
                        sys.stderr.flush()
                        try:
                            await _run_deeptalk(orchestrator)
                        except Exception:
                            _logger.exception("DeepTalk session error")
                        _logger.info("DeepTalk ended, restarting detector")
                        detector.start(loop)
                        activation_task = None
                        continue

                    response = await orchestrator.handle_interaction(transcript)
                    console.print(f"[bold white]You:[/] {transcript}")
                    console.print(f"[bold cyan]Nova:[/] {response}\n")

                except Exception:
                    logging.getLogger(__name__).exception(
                        "Voice interaction error"
                    )
                    console.print("[red]Terjadi kesalahan.[/]\n")

    except KeyboardInterrupt:
        pass
    finally:
        if activation_task and not activation_task.done():
            activation_task.cancel()
            try:
                await activation_task
            except asyncio.CancelledError:
                pass
        detector.stop()
        orchestrator.stop()
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

    # Start remote agent WebSocket server
    if config.remote_agent_enabled:
        try:
            from nova.remote.server import start_remote_server

            await start_remote_server()
            console.print(
                f"[dim]Remote agent server on ws://0.0.0.0:{config.remote_agent_port}[/]"
            )
        except Exception as e:
            logging.getLogger(__name__).warning("Remote agent server failed to start: %s", e)

    # Start Telegram bot (if configured)
    telegram_task = None
    if config.telegram_bot_token and config.telegram_allowed_users:
        try:
            from nova.messaging.telegram_bot import NovaTelegramBot

            tg_bot = NovaTelegramBot(
                token=config.telegram_bot_token,
                allowed_users=config.telegram_allowed_users,
                orchestrator=orchestrator,
            )
            telegram_task = asyncio.create_task(tg_bot.start())
            console.print("[dim]Telegram bot started[/]")
        except Exception as e:
            logging.getLogger(__name__).warning("Telegram bot failed to start: %s", e)

    # Start WhatsApp bridge client (if enabled)
    wa_client = None
    if config.whatsapp_enabled:
        try:
            from nova.messaging.whatsapp_client import NovaWhatsAppClient

            wa_client = NovaWhatsAppClient(
                orchestrator=orchestrator,
                allowed_numbers=config.whatsapp_allowed_jids or None,
            )
            await wa_client.start()
            console.print("[dim]WhatsApp client started (bridge at localhost:3001)[/]")
        except Exception as e:
            logging.getLogger(__name__).warning("WhatsApp client failed to start: %s", e)

    try:
        if args.text_only:
            await _text_mode(orchestrator)
        elif args.push_to_talk:
            await _voice_mode(orchestrator)
        elif args.hotkey:
            await _wake_word_mode(orchestrator, force_hotkey=True)
        else:
            await _wake_word_mode(orchestrator)
    except Exception:
        logging.getLogger(__name__).exception("NOVA terminated unexpectedly")
    finally:
        # Stop messaging services
        if telegram_task:
            telegram_task.cancel()
            try:
                await telegram_task
            except asyncio.CancelledError:
                pass
        if wa_client:
            await wa_client.stop()

        if config.remote_agent_enabled:
            try:
                from nova.remote.server import stop_remote_server

                await stop_remote_server()
            except Exception:
                pass
        console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")
        sys.stdout.flush()
        sys.stderr.flush()


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        console.print("\n[bold green]Sampai jumpa![/] (Goodbye!)")
    except Exception:
        logging.getLogger(__name__).exception("NOVA crashed")
        console.print("\n[red]NOVA crashed unexpectedly. Check logs.[/]")


if __name__ == "__main__":
    main()
