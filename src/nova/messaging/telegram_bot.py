"""Telegram bot for remote NOVA control.

Uses aiogram 3.x for async Telegram Bot API interaction.
Only responds to user IDs in the allowed list — all other messages
are silently ignored for security.
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

logger = logging.getLogger(__name__)


class NovaTelegramBot:
    """Telegram bot that routes messages through NOVA orchestrator.

    Security: Only responds to user IDs in the allowed list.
    All other messages are silently ignored.
    """

    def __init__(self, token: str, allowed_users: list[int], orchestrator) -> None:  # noqa: ANN001
        """Initialize the Telegram bot.

        Args:
            token: Telegram Bot API token from @BotFather.
            allowed_users: List of authorized Telegram user IDs.
            orchestrator: NOVA Orchestrator instance for processing messages.
        """
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._allowed = set(allowed_users)
        self._orchestrator = orchestrator
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Register message handlers with the dispatcher."""

        @self._dp.message(Command("start"))
        async def handle_start(message: types.Message) -> None:
            if not self._is_authorized(message):
                return
            await message.reply(
                "NOVA terhubung. Kirim pesan untuk mengontrol.\n\n"
                "Contoh:\n"
                "• nyalakan AC suhu 24\n"
                "• besok hujan gak?\n"
                "• matikan TV atas\n"
                "• ingatkan saya jam 3 meeting"
            )

        @self._dp.message(Command("status"))
        async def handle_status(message: types.Message) -> None:
            if not self._is_authorized(message):
                return
            await message.reply(
                "NOVA aktif.\n"
                "Voice: online\n"
                "IoT: connected"
            )

        @self._dp.message()
        async def handle_message(message: types.Message) -> None:
            if not self._is_authorized(message):
                return
            if not message.text:
                await message.reply("Saya hanya bisa memproses pesan teks.")
                return

            logger.info(
                "Telegram message from %s: '%s'",
                message.from_user.id,
                message.text,
            )

            try:
                response = await self._orchestrator.handle_interaction(
                    message.text,
                    mode="text",
                )
                await message.reply(response)
            except Exception:
                logger.exception("Error processing Telegram message")
                await message.reply("Terjadi kesalahan saat memproses pesan.")

    def _is_authorized(self, message: types.Message) -> bool:
        """Check if the sender is in the allowed users list.

        Args:
            message: Incoming Telegram message.

        Returns:
            True if the sender is authorized.
        """
        user = message.from_user
        if user is None or user.id not in self._allowed:
            uid = user.id if user else "unknown"
            uname = user.username if user else "unknown"
            logger.warning(
                "Unauthorized Telegram message from user %s (%s)", uid, uname,
            )
            return False
        return True

    async def start(self) -> None:
        """Start the bot polling loop. Run as asyncio task."""
        logger.info("Telegram bot starting...")
        try:
            await self._dp.start_polling(self._bot, handle_signals=False)
        except asyncio.CancelledError:
            logger.info("Telegram bot stopped")
        except Exception:
            logger.exception("Telegram bot crashed")
        finally:
            await self._bot.session.close()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        await self._dp.stop_polling()
        await self._bot.session.close()
