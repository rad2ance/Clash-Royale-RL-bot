from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adb_client import AdbClient


@dataclass
class BlueStacksInstance:
    """
    BlueStacks control surface.

    This class intentionally stays thin: use it as the adapter boundary
    where image parsing and policy decisions can be plugged in.
    """

    adb: AdbClient
    host: str = "127.0.0.1"
    port: int = 5555

    def connect(self) -> None:
        self.adb.connect(self.host, self.port)

    def take_screenshot(self, output_path: str | Path) -> None:
        self.adb.screencap(output_path)

    def play_card(self, card_slot: int, x: int, y: int, hand_slot_positions: dict[int, tuple[int, int]]) -> None:
        if card_slot not in hand_slot_positions:
            raise ValueError(f"Unknown card slot: {card_slot}")
        sx, sy = hand_slot_positions[card_slot]
        self.adb.tap(sx, sy)
        self.adb.tap(x, y)

