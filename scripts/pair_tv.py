"""Pair LG WebOS TV — run once per TV, accept popup on screen."""

import asyncio
import sys

from nova.iot.lg_webos import LGWebOSDriver

# 60 seconds to give user time to accept pairing on TV
_PAIR_TIMEOUT = 60


async def pair(ip: str, name: str) -> None:
    # Quick reachability check (TCP port 3000 = WebOS websocket)
    print(f"Checking if {name} ({ip}) is reachable...")
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, 3000), timeout=5,
        )
        writer.close()
        await writer.wait_closed()
        print("TV is reachable!")
    except (asyncio.TimeoutError, OSError) as e:
        print(f"GAGAL: TV tidak bisa dijangkau di {ip}:3000")
        print(f"  Error: {e}")
        print("  Pastikan:")
        print("  - TV menyala (bukan standby)")
        print("  - TV dan PC di jaringan Wi-Fi yang sama")
        print("  - IP address benar (cek di Settings > Network di TV)")
        return

    print(f"\nConnecting to {name}...")
    print(">> Accept popup di layar TV! (timeout 60 detik) <<\n")

    tv = LGWebOSDriver(ip=ip, name=name)
    try:
        await tv.connect(timeout=_PAIR_TIMEOUT)
    except ConnectionError as e:
        print(f"GAGAL: {e}")
        print("  Pastikan kamu sudah accept popup di TV.")
        return

    print(f"Paired! Simpan client_key ini ke .env:")
    if name == "tv_atas":
        print(f"  LG_TV_ATAS_CLIENT_KEY={tv.client_key}")
    elif name == "tv_bawah":
        print(f"  LG_TV_BAWAH_CLIENT_KEY={tv.client_key}")
    else:
        print(f"  Client key: {tv.client_key}")

    await tv.disconnect()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/pair_tv.py <IP> <NAME>")
        print("Example:")
        print("  python scripts/pair_tv.py 192.168.0.237 tv_atas")
        print("  python scripts/pair_tv.py 192.168.0.100 tv_bawah")
        sys.exit(1)

    asyncio.run(pair(sys.argv[1], sys.argv[2]))
