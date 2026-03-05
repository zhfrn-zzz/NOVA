"""Tests for IoT device control — AC (Tuya IR), TV Atas (IR+WebOS), TV Bawah (WebOS)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nova.iot.lg_webos  # noqa: F401 — ensure module is loaded for @patch
import nova.iot.tuya_cloud  # noqa: F401 — ensure module is loaded for @patch

# ── TuyaCloudDriver tests ────────────────────────────────────────────


class TestTuyaCloudDriver:
    """Test Tuya Cloud IR driver for AC and TV."""

    def _make_driver(self):
        from nova.iot.tuya_cloud import TuyaCloudDriver
        return TuyaCloudDriver(
            access_id="test_id", access_key="test_key", region="eu",
        )

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_power_on(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(power=True)

        assert "dinyalakan" in result
        mock_cloud._tuyaplatform.assert_called_once()
        call_args = mock_cloud._tuyaplatform.call_args
        assert "scenes/command" in call_args[0][0]  # combined endpoint
        assert call_args[1]["post"]["power"] == "1"

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_power_off(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(power=False)

        assert "dimatikan" in result

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_set_temp(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(temp=24)

        assert "24" in result
        call_args = mock_cloud._tuyaplatform.call_args
        assert "scenes/command" in call_args[0][0]
        assert call_args[1]["post"]["temp"] == "24"

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_set_mode(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(mode=0)

        assert "cool" in result
        call_args = mock_cloud._tuyaplatform.call_args
        assert "scenes/command" in call_args[0][0]
        assert call_args[1]["post"]["mode"] == "0"

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_set_fan(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(fan=2)

        assert "medium" in result

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_power_failure(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": False, "msg": "device offline"}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(power=True)

        assert "Gagal" in result

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_combined_commands(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command(power=True, temp=24)

        assert "dinyalakan" in result
        assert "24" in result
        assert mock_cloud._tuyaplatform.call_count == 1

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_ac_no_commands(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_ac_command()

        assert "Tidak ada" in result

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_tv_ir_command_success(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": True}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_tv_ir_command("Power")

        assert "berhasil" in result

    @patch("nova.iot.tuya_cloud.tinytuya.Cloud")
    @pytest.mark.asyncio
    async def test_tv_ir_command_failure(self, mock_cloud_cls):
        mock_cloud = MagicMock()
        mock_cloud._tuyaplatform.return_value = {"success": False}
        mock_cloud_cls.return_value = mock_cloud

        driver = self._make_driver()
        result = await driver.send_tv_ir_command("Power")

        assert "Gagal" in result

    def test_missing_credentials(self):
        from nova.iot.tuya_cloud import TuyaCloudDriver
        with patch.dict("os.environ", {"TUYA_ACCESS_ID": "", "TUYA_ACCESS_KEY": ""}):
            driver = TuyaCloudDriver(access_id="", access_key="")
            with pytest.raises(RuntimeError, match="belum diset"):
                driver._get_cloud()


# ── LGWebOSDriver tests ─────────────────────────────────────────────


class TestLGWebOSDriver:
    """Test LG WebOS TV driver."""

    def _make_driver(self):
        from nova.iot.lg_webos import LGWebOSDriver
        return LGWebOSDriver(ip="192.168.1.100", name="test_tv")

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.client_key = "saved_key"
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        result = await driver.connect()

        assert result is True
        mock_client.connect.assert_awaited_once()

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_connect_timeout(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.connect.side_effect = asyncio.TimeoutError()
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        with pytest.raises(ConnectionError, match="Tidak bisa terhubung"):
            await driver.connect()

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_launch_app_youtube(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.client_key = None
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        driver._client = mock_client  # Skip connect
        result = await driver.launch_app("youtube")

        assert "youtube" in result.lower()
        mock_client.launch_app.assert_awaited_once_with("youtube.leanback.v4")

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_launch_app_netflix(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        driver._client = mock_client
        result = await driver.launch_app("netflix")

        assert "netflix" in result.lower()
        mock_client.launch_app.assert_awaited_once_with("netflix")

    @pytest.mark.asyncio
    async def test_launch_app_unknown(self):
        driver = self._make_driver()
        driver._client = AsyncMock()
        driver._client.is_connected.return_value = True
        result = await driver.launch_app("nonexistent_app")

        assert "tidak dikenali" in result

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_set_volume(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        driver._client = mock_client
        result = await driver.set_volume(50)

        assert "50" in result
        mock_client.set_volume.assert_awaited_once_with(50)

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_volume_up(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        driver._client = mock_client
        result = await driver.volume_up()

        assert "dinaikkan" in result

    @patch("nova.iot.lg_webos.WebOsClient")
    @pytest.mark.asyncio
    async def test_power_off(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        driver = self._make_driver()
        driver._client = mock_client
        result = await driver.power_off()

        assert "dimatikan" in result
        mock_client.power_off.assert_awaited_once()


# ── control_device dispatch tests ────────────────────────────────────


class TestControlDevice:
    """Test the control_device tool dispatch logic."""

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        """Reset lazy-initialized driver singletons between tests."""
        import nova.tools.iot as iot_mod
        iot_mod._tuya_driver = None
        iot_mod._tv_atas_webos = None
        iot_mod._tv_bawah_webos = None
        yield
        iot_mod._tuya_driver = None
        iot_mod._tv_atas_webos = None
        iot_mod._tv_bawah_webos = None

    @pytest.mark.asyncio
    async def test_invalid_device(self):
        from nova.tools.iot import control_device
        result = await control_device("lamp", "on")
        assert "tidak dikenali" in result

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_on(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC dinyalakan."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        result = await control_device("ac", "on")

        assert "dinyalakan" in result
        mock_driver.send_ac_command.assert_awaited_once_with(power=True)

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_off(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC dimatikan."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        result = await control_device("ac", "off")

        assert "dimatikan" in result

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_set_temp(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC suhu 24°C."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("ac", "set_temp", "24")

        mock_driver.send_ac_command.assert_awaited_once_with(power=True, temp=24)

    @pytest.mark.asyncio
    async def test_ac_set_temp_out_of_range(self):
        from nova.tools.iot import control_device
        result = await control_device("ac", "set_temp", "50")
        assert "16-30" in result

    @pytest.mark.asyncio
    async def test_ac_set_temp_invalid(self):
        from nova.tools.iot import control_device
        result = await control_device("ac", "set_temp", "abc")
        assert "16-30" in result

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_set_mode_by_number(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC mode cool."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("ac", "set_mode", "0")
        mock_driver.send_ac_command.assert_awaited_once_with(power=True, mode=0)

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_set_mode_by_name(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC mode dingin."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("ac", "set_mode", "dingin")
        mock_driver.send_ac_command.assert_awaited_once_with(power=True, mode=0)

    @pytest.mark.asyncio
    async def test_ac_set_mode_invalid(self):
        from nova.tools.iot import control_device
        result = await control_device("ac", "set_mode", "turbo")
        assert "tidak valid" in result

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_ac_set_fan(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_ac_command.return_value = "AC kipas sedang."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("ac", "set_fan", "2")
        mock_driver.send_ac_command.assert_awaited_once_with(power=True, fan=2)

    @pytest.mark.asyncio
    async def test_ac_invalid_action(self):
        from nova.tools.iot import control_device
        result = await control_device("ac", "dance")
        assert "tidak dikenali" in result

    # ── TV Atas tests ────────────────────────────────────────────

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_tv_atas_power_on_uses_ir(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_tv_ir_command.return_value = "Perintah TV IR 'Power' berhasil dikirim."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("tv_atas", "on")

        mock_driver.send_tv_ir_command.assert_awaited_once_with("Power")

    @patch("nova.tools.iot.get_tv_atas_webos")
    @pytest.mark.asyncio
    async def test_tv_atas_off_uses_webos(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.power_off.return_value = "TV Atas dimatikan."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_atas", "off")

        assert "dimatikan" in result

    @patch("nova.tools.iot.get_tv_atas_webos")
    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_tv_atas_off_fallback_to_ir(self, mock_get_tuya, mock_get_webos):
        """If WebOS fails, fall back to IR for power off."""
        mock_webos = AsyncMock()
        mock_webos.power_off.side_effect = ConnectionError("TV unreachable")
        mock_get_webos.return_value = mock_webos

        mock_driver = AsyncMock()
        mock_driver.send_tv_ir_command.return_value = "Perintah TV IR 'Power' berhasil dikirim."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("tv_atas", "off")

        mock_driver.send_tv_ir_command.assert_awaited_once_with("Power")

    @patch("nova.tools.iot.get_tv_atas_webos")
    @pytest.mark.asyncio
    async def test_tv_atas_open_app(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.launch_app.return_value = "Membuka youtube di TV Atas."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_atas", "open_app", "youtube")

        assert "youtube" in result.lower()

    @patch("nova.tools.iot.get_tv_atas_webos")
    @pytest.mark.asyncio
    async def test_tv_atas_set_volume(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.set_volume.return_value = "Volume TV Atas diset ke 30."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_atas", "set_volume", "30")

        assert "30" in result

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_tv_atas_channel_up_uses_ir(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_tv_ir_command.return_value = "Perintah TV IR 'Channel+' berhasil."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("tv_atas", "channel_up")

        mock_driver.send_tv_ir_command.assert_awaited_once_with("Channel+")

    @patch("nova.tools.iot.get_tuya_driver")
    @pytest.mark.asyncio
    async def test_tv_atas_navigation_uses_ir(self, mock_get_tuya):
        mock_driver = AsyncMock()
        mock_driver.send_tv_ir_command.return_value = "Perintah TV IR 'OK' berhasil."
        mock_get_tuya.return_value = mock_driver

        from nova.tools.iot import control_device
        await control_device("tv_atas", "ok")

        mock_driver.send_tv_ir_command.assert_awaited_once_with("OK")

    # ── TV Bawah tests ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tv_bawah_power_on_blocked(self):
        from nova.tools.iot import control_device
        result = await control_device("tv_bawah", "on")
        assert "manual" in result.lower()

    @pytest.mark.asyncio
    async def test_tv_bawah_channel_blocked(self):
        from nova.tools.iot import control_device
        result = await control_device("tv_bawah", "channel_up")
        assert "tidak tersedia" in result

    @pytest.mark.asyncio
    async def test_tv_bawah_navigation_blocked(self):
        from nova.tools.iot import control_device
        for action in ("up", "down", "left", "right", "ok", "menu"):
            result = await control_device("tv_bawah", action)
            assert "tidak tersedia" in result

    @patch("nova.tools.iot.get_tv_bawah_webos")
    @pytest.mark.asyncio
    async def test_tv_bawah_off(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.power_off.return_value = "TV Bawah dimatikan."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_bawah", "off")

        assert "dimatikan" in result

    @patch("nova.tools.iot.get_tv_bawah_webos")
    @pytest.mark.asyncio
    async def test_tv_bawah_open_app(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.launch_app.return_value = "Membuka netflix di TV Bawah."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_bawah", "open_app", "netflix")

        assert "netflix" in result.lower()

    @patch("nova.tools.iot.get_tv_bawah_webos")
    @pytest.mark.asyncio
    async def test_tv_bawah_volume(self, mock_get_webos):
        mock_webos = AsyncMock()
        mock_webos.set_volume.return_value = "Volume TV Bawah diset ke 40."
        mock_get_webos.return_value = mock_webos

        from nova.tools.iot import control_device
        result = await control_device("tv_bawah", "set_volume", "40")

        assert "40" in result

    @pytest.mark.asyncio
    async def test_tv_bawah_no_ip_configured(self):
        from nova.tools.iot import control_device

        with patch.dict("os.environ", {}, clear=False):
            # Ensure LG_TV_BAWAH_IP is not set
            import os
            old = os.environ.pop("LG_TV_BAWAH_IP", None)
            try:
                result = await control_device("tv_bawah", "off")
                assert "belum dikonfigurasi" in result or "dimatikan" in result.lower()
            finally:
                if old is not None:
                    os.environ["LG_TV_BAWAH_IP"] = old

    # ── Volume validation ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_volume_out_of_range(self):
        from nova.tools.iot import control_device
        result = await control_device("tv_atas", "set_volume", "150")
        assert "0-100" in result

    @pytest.mark.asyncio
    async def test_volume_invalid(self):
        from nova.tools.iot import control_device
        result = await control_device("tv_atas", "set_volume", "loud")
        assert "0-100" in result


# ── Value parsing helper tests ───────────────────────────────────────


class TestValueParsing:
    """Test helper functions for value parsing."""

    def test_parse_int_valid(self):
        from nova.tools.iot import _parse_int
        assert _parse_int("24", 16, 30) == 24

    def test_parse_int_min_boundary(self):
        from nova.tools.iot import _parse_int
        assert _parse_int("16", 16, 30) == 16

    def test_parse_int_max_boundary(self):
        from nova.tools.iot import _parse_int
        assert _parse_int("30", 16, 30) == 30

    def test_parse_int_out_of_range(self):
        from nova.tools.iot import _parse_int
        assert _parse_int("35", 16, 30) is None

    def test_parse_int_not_a_number(self):
        from nova.tools.iot import _parse_int
        assert _parse_int("abc", 16, 30) is None

    def test_resolve_ac_mode_number(self):
        from nova.tools.iot import _resolve_ac_mode
        assert _resolve_ac_mode("0") == 0
        assert _resolve_ac_mode("4") == 4

    def test_resolve_ac_mode_name_id(self):
        from nova.tools.iot import _resolve_ac_mode
        assert _resolve_ac_mode("dingin") == 0
        assert _resolve_ac_mode("panas") == 1

    def test_resolve_ac_mode_name_en(self):
        from nova.tools.iot import _resolve_ac_mode
        assert _resolve_ac_mode("cool") == 0
        assert _resolve_ac_mode("heat") == 1

    def test_resolve_ac_mode_invalid(self):
        from nova.tools.iot import _resolve_ac_mode
        assert _resolve_ac_mode("turbo") is None

    def test_resolve_ac_fan_number(self):
        from nova.tools.iot import _resolve_ac_fan
        assert _resolve_ac_fan("0") == 0
        assert _resolve_ac_fan("3") == 3

    def test_resolve_ac_fan_name(self):
        from nova.tools.iot import _resolve_ac_fan
        assert _resolve_ac_fan("pelan") == 1
        assert _resolve_ac_fan("kencang") == 3

    def test_resolve_ac_fan_invalid(self):
        from nova.tools.iot import _resolve_ac_fan
        assert _resolve_ac_fan("max") is None


# ── Registry integration test ────────────────────────────────────────


class TestIoTInRegistry:
    """Verify control_device is registered correctly."""

    def test_control_device_in_tool_names(self):
        from nova.tools.registry import get_all_tool_names
        names = get_all_tool_names()
        assert "control_device" in names

    def test_control_device_declaration_exists(self):
        from nova.tools.registry import get_tool_declarations
        tools = get_tool_declarations()
        decls = tools[0].function_declarations
        names = [d.name for d in decls]
        assert "control_device" in names


# ── Action Reminder tests ─────────────────────────────────────────────


class TestActionReminders:
    """Test action reminder flow: store → retrieve → notify → execute."""

    def _make_store(self, tmp_path):
        from nova.memory.memory_store import MemoryStore
        return MemoryStore(db_path=str(tmp_path / "test.db"))

    def test_add_reminder_with_action_stores_json(self, tmp_path):
        """add_reminder stores action_json in DB when action is provided."""
        import json
        store = self._make_store(tmp_path)
        action = {"device": "tv_atas", "action": "off"}
        rid = store.add_reminder(
            message="matiin tv atas",
            remind_at="2099-01-01T00:00:00",
            action=action,
        )
        # Read raw from DB
        row = store._conn.execute(
            "SELECT action_json FROM reminders WHERE id = ?", (rid,)
        ).fetchone()
        assert row is not None
        assert json.loads(row["action_json"]) == action

    def test_add_reminder_without_action_stores_null(self, tmp_path):
        """add_reminder stores NULL action_json when no action provided."""
        store = self._make_store(tmp_path)
        rid = store.add_reminder(
            message="pengingat biasa",
            remind_at="2099-01-01T00:00:00",
        )
        row = store._conn.execute(
            "SELECT action_json FROM reminders WHERE id = ?", (rid,)
        ).fetchone()
        assert row["action_json"] is None

    def test_get_pending_reminders_includes_action(self, tmp_path):
        """get_pending_reminders returns action dict from stored action_json."""
        from datetime import datetime, timedelta
        store = self._make_store(tmp_path)
        action = {"device": "ac", "action": "off"}
        store.add_reminder(
            message="matiin AC",
            remind_at=(datetime.now() + timedelta(seconds=1)).isoformat(),
            lead_time=0,
            action=action,
        )
        now = datetime.now() + timedelta(seconds=2)
        pending = store.get_pending_reminders(now, window_minutes=0)
        assert len(pending) == 1
        assert pending[0]["action"] == action

    def test_get_pending_reminders_action_none_when_not_set(self, tmp_path):
        """get_pending_reminders returns action=None when not set."""
        from datetime import datetime, timedelta
        store = self._make_store(tmp_path)
        store.add_reminder(
            message="biasa",
            remind_at=(datetime.now() + timedelta(seconds=1)).isoformat(),
            lead_time=0,
        )
        now = datetime.now() + timedelta(seconds=2)
        pending = store.get_pending_reminders(now, window_minutes=0)
        assert len(pending) == 1
        assert pending[0]["action"] is None

    def test_notification_carries_action(self, tmp_path):
        """Scheduler passes action from reminder dict to Notification."""
        from datetime import datetime, timedelta

        from nova.heartbeat.queue import Notification, NotificationQueue, Urgency
        from nova.heartbeat.scheduler import HeartbeatScheduler
        from nova.memory.memory_store import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "test.db"))
        queue = NotificationQueue()
        scheduler = HeartbeatScheduler(memory_store=store, notification_queue=queue)

        action = {"device": "tv_bawah", "action": "off"}
        store.add_reminder(
            message="matiin tv bawah",
            remind_at=(datetime.now() + timedelta(seconds=1)).isoformat(),
            lead_time=0,
            action=action,
        )

        now = datetime.now() + timedelta(seconds=2)
        scheduler._check_reminders(now)

        assert queue.size() == 1
        notif = queue.get_next_urgent()
        assert notif is not None
        assert notif.action == action
        assert notif.urgency == Urgency.ACTIVE

    def test_notification_without_action_not_forced_active(self, tmp_path):
        """Regular reminders (no action) follow normal urgency — not forced ACTIVE."""
        from datetime import datetime, timedelta

        from nova.heartbeat.queue import NotificationQueue, Urgency
        from nova.heartbeat.scheduler import HeartbeatScheduler
        from nova.memory.memory_store import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "test.db"))
        queue = NotificationQueue()
        scheduler = HeartbeatScheduler(memory_store=store, notification_queue=queue)

        # remind_at = now + 15 min, lead_time = 20 → fires now (remind_at - lead = now-5)
        # time_until = 15 min → _calculate_dynamic_urgency returns GENTLE (not ACTIVE)
        now = datetime.now()
        remind_at = now + timedelta(minutes=15)
        store.add_reminder(
            message="biasa",
            remind_at=remind_at.isoformat(),
            lead_time=20,
        )

        scheduler._check_reminders(now)

        assert queue.size() == 1
        notif = queue.get_next_urgent()
        assert notif is not None
        assert notif.urgency == Urgency.GENTLE  # natural urgency, not forced ACTIVE

    @pytest.mark.asyncio
    async def test_set_reminder_tool_with_action(self, tmp_path):
        """set_reminder tool passes action through to memory store."""
        import json

        from nova.memory.memory_store import MemoryStore

        store = MemoryStore(db_path=str(tmp_path / "test.db"))
        with patch("nova.tools.heartbeat_reminders.get_memory_store", return_value=store):
            from nova.tools.heartbeat_reminders import set_reminder
            action = {"device": "tv_atas", "action": "off"}
            result = await set_reminder(
                message="matiin tv atas",
                delay_minutes=20,
                action=action,
            )

        assert "tv_atas" in result
        assert "off" in result
        # Verify DB
        row = store._conn.execute(
            "SELECT action_json FROM reminders ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert json.loads(row["action_json"]) == action
