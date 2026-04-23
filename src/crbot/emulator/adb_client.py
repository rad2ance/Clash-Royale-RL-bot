from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdbClient:
    """Minimal ADB wrapper used by BlueStacks integration."""

    adb_path: str = "adb"
    device_serial: str | None = None

    def _base_cmd(self) -> list[str]:
        cmd = [self.adb_path]
        if self.device_serial:
            cmd += ["-s", self.device_serial]
        return cmd

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = self._base_cmd() + list(args)
        return subprocess.run(cmd, check=check, capture_output=True, text=True)

    def connect(self, host: str, port: int) -> None:
        self.run("connect", f"{host}:{port}")

    def tap(self, x: int, y: int) -> None:
        self.run("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self.run(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def screencap(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_remote = "/sdcard/__crbot_screen.png"
        self.run("shell", "screencap", "-p", tmp_remote)
        self.run("pull", tmp_remote, str(out))
        self.run("shell", "rm", tmp_remote)

