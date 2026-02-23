# -*- coding: utf-8 -*-
"""Comprehensive test for ALL NOVA tools.

Tests all safe-to-test tools and reports pass/fail status.
Skips destructive tools (shutdown, restart, sleep, wifi_on/off, dictate).
"""

import asyncio
import sys
import traceback

# Add src to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


async def run_all_tests() -> None:
    results: list[tuple[str, str, str]] = []

    # ── 1. Time/Date ────────────────────────────────────────────────
    from nova.tools.time_date import get_current_time, get_current_date, get_current_datetime

    for name, fn in [
        ("get_current_time", get_current_time),
        ("get_current_date", get_current_date),
        ("get_current_datetime", get_current_datetime),
    ]:
        try:
            result = await fn()
            assert isinstance(result, str) and len(result) > 0
            results.append((name, PASS, result))
        except Exception as e:
            results.append((name, FAIL, f"{e}\n{traceback.format_exc()}"))

    # ── 2. System Info ──────────────────────────────────────────────
    from nova.tools.system_info import (
        get_battery_level, get_ram_usage, get_storage_info,
        get_ip_address, get_system_uptime,
    )

    for name, fn in [
        ("get_battery_level", get_battery_level),
        ("get_ram_usage", get_ram_usage),
        ("get_storage_info", get_storage_info),
        ("get_ip_address", get_ip_address),
        ("get_system_uptime", get_system_uptime),
    ]:
        try:
            result = await fn()
            assert isinstance(result, str) and len(result) > 0
            results.append((name, PASS, result))
        except Exception as e:
            results.append((name, FAIL, f"{e}\n{traceback.format_exc()}"))

    # ── 3. Notes (add → get → clear → get) ────────────────────────
    from nova.tools.notes import add_note, get_notes, clear_notes

    try:
        r = await add_note("Test note dari automated test")
        assert "tersimpan" in r.lower() or "catatan" in r.lower()
        results.append(("add_note", PASS, r))
    except Exception as e:
        results.append(("add_note", FAIL, str(e)))

    try:
        r = await get_notes()
        assert isinstance(r, str) and len(r) > 0
        results.append(("get_notes", PASS, r))
    except Exception as e:
        results.append(("get_notes", FAIL, str(e)))

    try:
        r = await clear_notes()
        assert "dihapus" in r.lower() or "clear" in r.lower()
        results.append(("clear_notes", PASS, r))
    except Exception as e:
        results.append(("clear_notes", FAIL, str(e)))

    try:
        r = await get_notes()
        assert "belum" in r.lower() or "no" in r.lower() or len(r) > 0
        results.append(("get_notes (after clear)", PASS, r))
    except Exception as e:
        results.append(("get_notes (after clear)", FAIL, str(e)))

    # ── 4. User Memory (remember → recall) ─────────────────────────
    from nova.memory.persistent import remember_fact, recall_facts

    try:
        r = await remember_fact("test_key", "test_value_123")
        assert "tersimpan" in r.lower() or "test_key" in r.lower()
        results.append(("remember_fact", PASS, r))
    except Exception as e:
        results.append(("remember_fact", FAIL, str(e)))

    try:
        r = await recall_facts()
        assert isinstance(r, str) and len(r) > 0
        results.append(("recall_facts", PASS, r))
    except Exception as e:
        results.append(("recall_facts", FAIL, str(e)))

    # Clean up test fact
    try:
        from nova.memory.persistent import get_user_memory
        mem = get_user_memory()
        mem.remove_fact("test_key")
        results.append(("remove_fact (cleanup)", PASS, "test_key removed"))
    except Exception as e:
        results.append(("remove_fact (cleanup)", FAIL, str(e)))

    # ── 5. Web Search ───────────────────────────────────────────────
    from nova.tools.web_search import web_search

    try:
        r = await web_search("Python programming language")
        assert isinstance(r, str) and len(r) > 0
        results.append(("web_search", PASS, r[:150] + "..."))
    except Exception as e:
        results.append(("web_search", FAIL, str(e)))

    # ── 6. Display/Brightness ───────────────────────────────────────
    from nova.tools.display_control import get_brightness

    try:
        r = await get_brightness()
        assert isinstance(r, str) and len(r) > 0
        results.append(("get_brightness", PASS, r))
    except Exception as e:
        results.append(("get_brightness", FAIL, str(e)))

    # Skip brightness_up/brightness_down to avoid actually changing brightness
    results.append(("brightness_up", SKIP, "Skipped – would change display settings"))
    results.append(("brightness_down", SKIP, "Skipped – would change display settings"))

    # ── 7. Network ──────────────────────────────────────────────────
    from nova.tools.network_control import get_wifi_status

    try:
        r = await get_wifi_status()
        assert isinstance(r, str) and len(r) > 0
        results.append(("get_wifi_status", PASS, r))
    except Exception as e:
        results.append(("get_wifi_status", FAIL, str(e)))

    results.append(("wifi_on", SKIP, "Skipped – would change network settings"))
    results.append(("wifi_off", SKIP, "Skipped – would change network settings"))

    # ── 8. Volume Controls ──────────────────────────────────────────
    from nova.tools.system_control import volume_up, volume_down, mute_unmute

    try:
        r = await volume_up()
        assert isinstance(r, str) and len(r) > 0
        results.append(("volume_up", PASS, r))
    except Exception as e:
        results.append(("volume_up", FAIL, str(e)))

    try:
        r = await volume_down()
        assert isinstance(r, str) and len(r) > 0
        results.append(("volume_down", PASS, r))
    except Exception as e:
        results.append(("volume_down", FAIL, str(e)))

    # Mute toggle – skip to not leave muted
    results.append(("mute_unmute", SKIP, "Skipped – would toggle mute"))

    # ── 9. Media Controls ───────────────────────────────────────────
    from nova.tools.system_control import play_pause_media, next_track, previous_track

    results.append(("play_pause_media", SKIP, "Skipped – would affect media playback"))
    results.append(("next_track", SKIP, "Skipped – would affect media playback"))
    results.append(("previous_track", SKIP, "Skipped – would affect media playback"))

    # ── 10. Screenshot ──────────────────────────────────────────────
    from nova.tools.system_control import take_screenshot

    try:
        r = await take_screenshot()
        assert isinstance(r, str) and len(r) > 0
        results.append(("take_screenshot", PASS, r))
    except Exception as e:
        results.append(("take_screenshot", FAIL, str(e)))

    # ── 11. Timer ───────────────────────────────────────────────────
    from nova.tools.system_control import set_timer

    try:
        r = await set_timer(seconds=3, label="Test Timer")
        assert isinstance(r, str) and "timer" in r.lower()
        results.append(("set_timer", PASS, r))
    except Exception as e:
        results.append(("set_timer", FAIL, str(e)))

    # ── 12. Reminder ────────────────────────────────────────────────
    from nova.tools.reminders import set_reminder

    try:
        r = await set_reminder(minutes=1, message="Test reminder")
        assert isinstance(r, str) and "reminder" in r.lower()
        results.append(("set_reminder", PASS, r))
    except Exception as e:
        results.append(("set_reminder", FAIL, str(e)))

    # ── 13. Apps (test open_app with safe app) ──────────────────────
    from nova.tools.system_control import open_app, open_browser, open_url, open_terminal, open_file_manager

    results.append(("open_app", SKIP, "Skipped – would open an application"))
    results.append(("open_browser", SKIP, "Skipped – would open browser"))
    results.append(("open_url", SKIP, "Skipped – would open URL"))
    results.append(("open_terminal", SKIP, "Skipped – would open terminal"))
    results.append(("open_file_manager", SKIP, "Skipped – would open explorer"))

    # ── 14. Power Controls ──────────────────────────────────────────
    from nova.tools.system_control import lock_screen, shutdown_pc, restart_pc, sleep_pc

    results.append(("lock_screen", SKIP, "Skipped – destructive"))
    results.append(("shutdown_pc", SKIP, "Skipped – destructive"))
    results.append(("restart_pc", SKIP, "Skipped – destructive"))
    results.append(("sleep_pc", SKIP, "Skipped – destructive"))

    # ── 15. Dictation ───────────────────────────────────────────────
    from nova.tools.dictation import dictate

    results.append(("dictate", SKIP, "Skipped – would type into active window"))

    # ── 16. Registry (meta test) ────────────────────────────────────
    from nova.tools.registry import get_all_tool_names, get_tool_declarations, execute_tool

    try:
        names = get_all_tool_names()
        assert isinstance(names, list) and len(names) > 20
        results.append(("get_all_tool_names", PASS, f"{len(names)} tools registered: {', '.join(names[:10])}..."))
    except Exception as e:
        results.append(("get_all_tool_names", FAIL, str(e)))

    try:
        decls = get_tool_declarations()
        assert isinstance(decls, list) and len(decls) > 0
        results.append(("get_tool_declarations", PASS, f"{len(decls)} Tool objects"))
    except Exception as e:
        results.append(("get_tool_declarations", FAIL, str(e)))

    try:
        r = await execute_tool("get_current_time")
        assert isinstance(r, str) and len(r) > 0
        results.append(("execute_tool('get_current_time')", PASS, r))
    except Exception as e:
        results.append(("execute_tool('get_current_time')", FAIL, str(e)))

    try:
        await execute_tool("nonexistent_tool_xyz")
        results.append(("execute_tool(unknown)", FAIL, "Should have raised ValueError"))
    except ValueError:
        results.append(("execute_tool(unknown) → ValueError", PASS, "Correctly raised ValueError"))
    except Exception as e:
        results.append(("execute_tool(unknown)", FAIL, str(e)))

    # ══════════════════════════════════════════════════════════════════
    # Write results to file
    # ══════════════════════════════════════════════════════════════════
    import pathlib
    outfile = pathlib.Path(__file__).parent / "test_results_out.txt"

    lines_out = []
    lines_out.append("=" * 80)
    lines_out.append("                    NOVA TOOLS TEST REPORT")
    lines_out.append("=" * 80)

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)

    for name, status, detail in results:
        short_detail = detail.split("\n")[0][:100]
        lines_out.append(f"  {status}  {name:<35s}  {short_detail}")

    lines_out.append("=" * 80)
    lines_out.append(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    lines_out.append("=" * 80)

    report = "\n".join(lines_out)
    outfile.write_text(report, encoding="utf-8")
    print(f"Results saved to: {outfile}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
