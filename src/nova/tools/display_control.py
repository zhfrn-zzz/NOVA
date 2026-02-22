"""Display brightness control tool — adjust screen brightness.

Uses screen_brightness_control library for cross-platform brightness management.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def brightness_up() -> str:
    """Increase screen brightness by 10%.

    Returns:
        Status message with current brightness level.
    """
    try:
        import screen_brightness_control as sbc

        current = await asyncio.to_thread(sbc.get_brightness)
        # sbc.get_brightness() returns a list of values (one per display)
        level = current[0] if isinstance(current, list) else current
        new_level = min(level + 10, 100)
        await asyncio.to_thread(sbc.set_brightness, new_level)
        logger.info("Tool brightness_up → %d%%", new_level)
        return f"Brightness dinaikkan ke {new_level}%."
    except ImportError:
        return (
            "screen-brightness-control belum terinstall. "
            "Jalankan: pip install screen-brightness-control"
        )
    except Exception as e:
        logger.error("Failed to increase brightness: %s", e)
        return f"Gagal menaikkan brightness: {e}"


async def brightness_down() -> str:
    """Decrease screen brightness by 10%.

    Returns:
        Status message with current brightness level.
    """
    try:
        import screen_brightness_control as sbc

        current = await asyncio.to_thread(sbc.get_brightness)
        level = current[0] if isinstance(current, list) else current
        new_level = max(level - 10, 0)
        await asyncio.to_thread(sbc.set_brightness, new_level)
        logger.info("Tool brightness_down → %d%%", new_level)
        return f"Brightness diturunkan ke {new_level}%."
    except ImportError:
        return (
            "screen-brightness-control belum terinstall. "
            "Jalankan: pip install screen-brightness-control"
        )
    except Exception as e:
        logger.error("Failed to decrease brightness: %s", e)
        return f"Gagal menurunkan brightness: {e}"


async def get_brightness() -> str:
    """Get the current screen brightness level.

    Returns:
        Current brightness percentage.
    """
    try:
        import screen_brightness_control as sbc

        current = await asyncio.to_thread(sbc.get_brightness)
        level = current[0] if isinstance(current, list) else current
        logger.info("Tool get_brightness → %d%%", level)
        return f"Brightness saat ini: {level}%."
    except ImportError:
        return (
            "screen-brightness-control belum terinstall. "
            "Jalankan: pip install screen-brightness-control"
        )
    except Exception as e:
        logger.error("Failed to get brightness: %s", e)
        return f"Gagal mendapatkan brightness: {e}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== Display Control Tools Test ===")
        print(await get_brightness())
        print(await brightness_up())
        print(await get_brightness())
        print(await brightness_down())
        print(await get_brightness())

    asyncio.run(main())
