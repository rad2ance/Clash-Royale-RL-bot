from __future__ import annotations

import argparse
from pathlib import Path

from crbot.emulator import AdbClient, BlueStacksInstance


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple BlueStacks/ADB smoke test.")
    parser.add_argument("--adb-path", type=str, default="adb")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--screenshot", type=str, default="artifacts/bluestacks_smoke.png")
    args = parser.parse_args()

    adb = AdbClient(adb_path=args.adb_path)
    bs = BlueStacksInstance(adb=adb, host=args.host, port=args.port)
    bs.connect()

    out = Path(args.screenshot)
    bs.take_screenshot(out)
    print(f"[done] screenshot captured: {out.resolve()}")


if __name__ == "__main__":
    main()

