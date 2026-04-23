from __future__ import annotations

import re
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

    def shell_stream(self, *args: str) -> subprocess.Popen[str]:
        cmd = self._base_cmd() + ["shell", *args]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

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

    def find_touch_input_device(self) -> str | None:
        output = self.run("shell", "getevent", "-lp").stdout
        current_device: str | None = None
        has_x = False
        has_y = False
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("add device"):
                if current_device and has_x and has_y:
                    return current_device
                m = re.search(r"(/dev/input/event\d+)", line)
                current_device = m.group(1) if m else None
                has_x = False
                has_y = False
                continue
            if current_device is None:
                continue
            if "ABS_MT_POSITION_X" in line or re.search(r"\bABS_X\b", line):
                has_x = True
            if "ABS_MT_POSITION_Y" in line or re.search(r"\bABS_Y\b", line):
                has_y = True
        if current_device and has_x and has_y:
            return current_device
        return None

    def get_touch_axis_max(self, input_device: str) -> tuple[int, int]:
        output = self.run("shell", "getevent", "-lp", input_device).stdout
        max_x: int | None = None
        max_y: int | None = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            max_match = re.search(r"\bmax\s+(-?\d+)", line)
            if not max_match:
                continue
            max_val = int(max_match.group(1))
            if "ABS_MT_POSITION_X" in line or re.search(r"\bABS_X\b", line):
                max_x = max_val
            if "ABS_MT_POSITION_Y" in line or re.search(r"\bABS_Y\b", line):
                max_y = max_val
        return (max_x or 32767, max_y or 32767)
