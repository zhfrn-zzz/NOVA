================================================================================
                    NOVA TOOLS TEST REPORT
================================================================================
  [PASS]  get_current_time                     16:53
  [PASS]  get_current_date                     Senin, 23 Februari 2026
  [PASS]  get_current_datetime                 Senin, 23 Februari 2026, pukul 16:53
  [PASS]  get_battery_level                    Baterai: 70%, tidak mengisi (discharging).
  [PASS]  get_ram_usage                        RAM: 12.0 GB / 31.8 GB (37.6% terpakai).
  [PASS]  get_storage_info                     Storage: 96.9 GB / 953.0 GB terpakai (856.1 GB tersisa, 10.2%).
  [PASS]  get_ip_address                       IP lokal: 192.168.0.76, IP publik: 111.95.24.89.
  [PASS]  get_system_uptime                    Sistem sudah menyala selama 10 menit.
  [PASS]  add_note                             Catatan tersimpan: Test note dari automated test
  [PASS]  get_notes                            Catatan terakhir:
  [PASS]  clear_notes                          Semua catatan telah dihapus.
  [PASS]  get_notes (after clear)              Belum ada catatan.
  [PASS]  remember_fact                        Tersimpan: test_key=test_value_123
  [PASS]  recall_facts                         User facts: name=ZHAFRAN, preferred_browser=Google Chrome, test_key=test_value_123
  [PASS]  remove_fact (cleanup)                test_key removed
  [PASS]  web_search                           1. Python (programming language): Python is a high-level, general-purpose programming language. Its 
  [PASS]  get_brightness                       Brightness saat ini: 84%.
  [SKIP]  brightness_up                        Skipped – would change display settings
  [SKIP]  brightness_down                      Skipped – would change display settings
  [PASS]  get_wifi_status                      Wi-Fi terhubung ke: N5F.
  [SKIP]  wifi_on                              Skipped – would change network settings
  [SKIP]  wifi_off                             Skipped – would change network settings
  [PASS]  volume_up                            Volume telah dinaikkan.
  [PASS]  volume_down                          Volume telah diturunkan.
  [SKIP]  mute_unmute                          Skipped – would toggle mute
  [SKIP]  play_pause_media                     Skipped – would affect media playback
  [SKIP]  next_track                           Skipped – would affect media playback
  [SKIP]  previous_track                       Skipped – would affect media playback
  [PASS]  take_screenshot                      Screenshot tersimpan di C:\Users\Zhafran\Pictures\Screenshots\screenshot_20260223_165324.png.
  [PASS]  set_timer                            Timer Test Timer telah diset untuk 3 detik.
  [PASS]  set_reminder                         Reminder diset untuk 1 menit lagi: Test reminder
  [SKIP]  open_app                             Skipped – would open an application
  [SKIP]  open_browser                         Skipped – would open browser
  [SKIP]  open_url                             Skipped – would open URL
  [SKIP]  open_terminal                        Skipped – would open terminal
  [SKIP]  open_file_manager                    Skipped – would open explorer
  [SKIP]  lock_screen                          Skipped – destructive
  [SKIP]  shutdown_pc                          Skipped – destructive
  [SKIP]  restart_pc                           Skipped – destructive
  [SKIP]  sleep_pc                             Skipped – destructive
  [SKIP]  dictate                              Skipped – would type into active window
  [PASS]  get_all_tool_names                   39 tools registered: add_note, brightness_down, brightness_up, clear_notes, dictate, get_battery_lev
  [PASS]  get_tool_declarations                1 Tool objects
  [PASS]  execute_tool('get_current_time')     16:53
  [PASS]  execute_tool(unknown) → ValueError   Correctly raised ValueError
================================================================================
  Total: 45 | Passed: 27 | Failed: 0 | Skipped: 18
================================================================================