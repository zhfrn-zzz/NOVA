"""Time and date tools — answers time/date queries without cloud API calls."""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def get_current_time() -> str:
    """Get the current local time as a human-readable string.

    Returns:
        Formatted time string, e.g. "14:30" or "2:30 PM".
    """
    now = datetime.now()
    result = now.strftime("%H:%M")
    logger.info("Tool get_current_time → %s", result)
    return result


async def get_current_date() -> str:
    """Get the current local date as a human-readable string.

    Returns:
        Formatted date string, e.g. "Sabtu, 22 Februari 2026".
    """
    now = datetime.now()
    # Use locale-independent format: Day, DD Month YYYY
    months_id = [
        "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    days_id = [
        "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu",
    ]
    day_name = days_id[now.weekday()]
    month_name = months_id[now.month]
    result = f"{day_name}, {now.day} {month_name} {now.year}"
    logger.info("Tool get_current_date → %s", result)
    return result


async def get_current_datetime() -> str:
    """Get the current local date and time as a human-readable string.

    Returns:
        Formatted datetime string combining date and time.
    """
    date_str = await get_current_date()
    time_str = await get_current_time()
    result = f"{date_str}, pukul {time_str}"
    logger.info("Tool get_current_datetime → %s", result)
    return result


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=== Time/Date Tools Test ===")
        print(f"Time: {await get_current_time()}")
        print(f"Date: {await get_current_date()}")
        print(f"DateTime: {await get_current_datetime()}")

    asyncio.run(main())
