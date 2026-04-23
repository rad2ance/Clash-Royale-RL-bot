from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from crbot.data import EpisodeBatch


@dataclass(frozen=True)
class FrameRecord:
    timestamp: float
    frame_path: str


@dataclass(frozen=True)
class TapEvent:
    timestamp: float
    raw_x: int
    raw_y: int
    screen_x: int
    screen_y: int


@dataclass(frozen=True)
class ActionLabel:
    """
    Generic action label tied to timestamped frames.

    Preferred format is `action` (already encoded discrete action id).
    Alternative format is (`slot`, `grid_x`, `grid_y`) and action is encoded.
    """

    timestamp: float
    action: int | None = None
    slot: int | None = None
    grid_x: int | None = None
    grid_y: int | None = None


@dataclass(frozen=True)
class UiLayout:
    """
    Normalized UI layout (0..1 relative coordinates).

    Defaults target a portrait Clash Royale battle layout and should be
    calibrated per emulator profile for best quality.
    """

    hand_left: float = 0.03
    hand_right: float = 0.97
    hand_top: float = 0.80
    hand_bottom: float = 0.98
    arena_left: float = 0.02
    arena_right: float = 0.98
    arena_top: float = 0.08
    arena_bottom: float = 0.79
    hand_slots: int = 4
    action_pair_timeout_s: float = 2.0

    def hand_slot_for_point(self, x: int, y: int, width: int, height: int) -> int | None:
        nx = x / max(width, 1)
        ny = y / max(height, 1)
        if nx < self.hand_left or nx > self.hand_right or ny < self.hand_top or ny > self.hand_bottom:
            return None
        slot_w = (self.hand_right - self.hand_left) / self.hand_slots
        rel = (nx - self.hand_left) / max(slot_w, 1e-6)
        slot = int(rel)
        return max(0, min(self.hand_slots - 1, slot))

    def arena_contains(self, x: int, y: int, width: int, height: int) -> bool:
        nx = x / max(width, 1)
        ny = y / max(height, 1)
        return self.arena_left <= nx <= self.arena_right and self.arena_top <= ny <= self.arena_bottom


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_frame_records(path: str | Path) -> list[FrameRecord]:
    return [FrameRecord(**row) for row in load_jsonl(path)]


def load_tap_events(path: str | Path) -> list[TapEvent]:
    return [TapEvent(**row) for row in load_jsonl(path)]


def load_action_labels(path: str | Path) -> list[ActionLabel]:
    labels: list[ActionLabel] = []
    for row in load_jsonl(path):
        labels.append(
            ActionLabel(
                timestamp=float(row["timestamp"]),
                action=None if row.get("action") is None else int(row.get("action")),
                slot=None if row.get("slot") is None else int(row.get("slot")),
                grid_x=None if row.get("grid_x") is None else int(row.get("grid_x")),
                grid_y=None if row.get("grid_y") is None else int(row.get("grid_y")),
            )
        )
    return labels


GETEVENT_RE = re.compile(
    r"\[\s*(?P<ts>\d+\.\d+)\]\s+(?P<dev>/dev/input/event\d+):\s+(?P<etype>\S+)\s+(?P<ecode>\S+)\s+(?P<value>\S+)"
)


@dataclass
class TouchTracker:
    x: int | None = None
    y: int | None = None
    down: bool = False


def _parse_input_value(token: str) -> int:
    t = token.strip().lower()
    if t in {"down", "up"}:
        return 1 if t == "down" else 0
    if t == "ffffffff":
        return -1
    if t.startswith("0x"):
        return int(t, 16)
    if re.fullmatch(r"[0-9a-f]+", t):
        if any(ch.isalpha() for ch in t):
            return int(t, 16)
        # getevent often prints hex as fixed-width zero-padded numbers.
        if len(t) > 4 and t.startswith("0"):
            return int(t, 16)
        return int(t, 10)
    try:
        return int(t, 10)
    except ValueError:
        return int(t, 16)


def scale_touch_to_screen(raw_x: int, raw_y: int, max_x: int, max_y: int, width: int, height: int) -> tuple[int, int]:
    sx = int(np.clip(round(raw_x / max(max_x, 1) * (width - 1)), 0, width - 1))
    sy = int(np.clip(round(raw_y / max(max_y, 1) * (height - 1)), 0, height - 1))
    return sx, sy


def parse_getevent_line(
    line: str,
    tracker: TouchTracker,
    screen_width: int,
    screen_height: int,
    max_x: int,
    max_y: int,
) -> TapEvent | None:
    m = GETEVENT_RE.search(line)
    if not m:
        return None
    ts = float(m.group("ts"))
    ecode = m.group("ecode")
    value = _parse_input_value(m.group("value"))

    if ecode in {"ABS_MT_POSITION_X", "ABS_X"}:
        tracker.x = value
        return None
    if ecode in {"ABS_MT_POSITION_Y", "ABS_Y"}:
        tracker.y = value
        return None

    is_down_signal = ecode == "BTN_TOUCH" and value == 1
    is_up_signal = (ecode == "BTN_TOUCH" and value == 0) or (ecode == "ABS_MT_TRACKING_ID" and value == -1)

    if is_down_signal:
        tracker.down = True
        return None

    if is_up_signal:
        if tracker.down and tracker.x is not None and tracker.y is not None:
            sx, sy = scale_touch_to_screen(tracker.x, tracker.y, max_x, max_y, screen_width, screen_height)
            tracker.down = False
            return TapEvent(
                timestamp=ts,
                raw_x=int(tracker.x),
                raw_y=int(tracker.y),
                screen_x=sx,
                screen_y=sy,
            )
        tracker.down = False
    return None


def screen_to_grid(
    x: int,
    y: int,
    width: int,
    height: int,
    layout: UiLayout,
    grid_w: int,
    grid_h: int,
) -> tuple[int, int]:
    nx = x / max(width, 1)
    ny = y / max(height, 1)
    rel_x = (nx - layout.arena_left) / max(layout.arena_right - layout.arena_left, 1e-6)
    rel_y = (ny - layout.arena_top) / max(layout.arena_bottom - layout.arena_top, 1e-6)
    gx = int(np.clip(np.floor(rel_x * grid_w), 0, grid_w - 1))
    gy = int(np.clip(np.floor(rel_y * grid_h), 0, grid_h - 1))
    return gx, gy


def encode_action_from_slot_and_grid(slot: int, grid_x: int, grid_y: int, hand_size: int, grid_w: int, grid_h: int) -> int:
    if slot < 0 or slot >= hand_size:
        raise ValueError(f"slot out of range: {slot}")
    if grid_x < 0 or grid_x >= grid_w or grid_y < 0 or grid_y >= grid_h:
        raise ValueError(f"grid out of range: ({grid_x}, {grid_y})")
    return 1 + slot * (grid_w * grid_h) + grid_y * grid_w + grid_x


def _closest_frame_before(frames: list[FrameRecord], ts: float) -> FrameRecord | None:
    selected: FrameRecord | None = None
    for frame in frames:
        if frame.timestamp <= ts:
            selected = frame
        else:
            break
    return selected


def _encode_frame_image(frame_path: str | Path, resize_w: int, resize_h: int) -> np.ndarray:
    with Image.open(frame_path) as im:
        arr = np.array(im.convert("L").resize((resize_w, resize_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return arr.reshape(-1).astype(np.float32)


def build_episode_from_logs(
    frames: list[FrameRecord],
    taps: list[TapEvent],
    screen_width: int,
    screen_height: int,
    layout: UiLayout,
    hand_size: int = 4,
    grid_w: int = 8,
    grid_h: int = 14,
    resize_w: int = 96,
    resize_h: int = 54,
) -> EpisodeBatch | None:
    frames = sorted(frames, key=lambda x: x.timestamp)
    taps = sorted(taps, key=lambda x: x.timestamp)
    pending_slot: int | None = None
    pending_slot_ts: float | None = None

    obs: list[np.ndarray] = []
    acts: list[int] = []
    rews: list[float] = []
    dones: list[bool] = []

    for tap in taps:
        slot = layout.hand_slot_for_point(tap.screen_x, tap.screen_y, screen_width, screen_height)
        if slot is not None:
            pending_slot = slot
            pending_slot_ts = tap.timestamp
            continue

        if pending_slot is None or pending_slot_ts is None:
            continue
        if tap.timestamp - pending_slot_ts > layout.action_pair_timeout_s:
            pending_slot = None
            pending_slot_ts = None
            continue
        if not layout.arena_contains(tap.screen_x, tap.screen_y, screen_width, screen_height):
            continue

        grid_x, grid_y = screen_to_grid(
            x=tap.screen_x,
            y=tap.screen_y,
            width=screen_width,
            height=screen_height,
            layout=layout,
            grid_w=grid_w,
            grid_h=grid_h,
        )
        action = encode_action_from_slot_and_grid(
            slot=pending_slot,
            grid_x=grid_x,
            grid_y=grid_y,
            hand_size=hand_size,
            grid_w=grid_w,
            grid_h=grid_h,
        )
        frame = _closest_frame_before(frames, tap.timestamp)
        if frame is None:
            continue
        obs.append(_encode_frame_image(frame.frame_path, resize_w=resize_w, resize_h=resize_h))
        acts.append(action)
        rews.append(0.0)
        dones.append(False)
        pending_slot = None
        pending_slot_ts = None

    if not obs:
        return None
    dones[-1] = True
    return EpisodeBatch(
        observations=np.stack(obs, axis=0).astype(np.float32),
        actions=np.array(acts, dtype=np.int64),
        rewards=np.array(rews, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def build_episode_from_frame_actions(
    frames: list[FrameRecord],
    labels: list[ActionLabel],
    hand_size: int = 4,
    grid_w: int = 8,
    grid_h: int = 14,
    resize_w: int = 96,
    resize_h: int = 54,
) -> EpisodeBatch | None:
    frames = sorted(frames, key=lambda x: x.timestamp)
    labels = sorted(labels, key=lambda x: x.timestamp)

    obs: list[np.ndarray] = []
    acts: list[int] = []
    rews: list[float] = []
    dones: list[bool] = []

    for label in labels:
        action: int
        if label.action is not None:
            action = int(label.action)
        else:
            if label.slot is None or label.grid_x is None or label.grid_y is None:
                continue
            action = encode_action_from_slot_and_grid(
                slot=int(label.slot),
                grid_x=int(label.grid_x),
                grid_y=int(label.grid_y),
                hand_size=hand_size,
                grid_w=grid_w,
                grid_h=grid_h,
            )
        frame = _closest_frame_before(frames, label.timestamp)
        if frame is None:
            continue
        obs.append(_encode_frame_image(frame.frame_path, resize_w=resize_w, resize_h=resize_h))
        acts.append(action)
        rews.append(0.0)
        dones.append(False)

    if not obs:
        return None
    dones[-1] = True
    return EpisodeBatch(
        observations=np.stack(obs, axis=0).astype(np.float32),
        actions=np.array(acts, dtype=np.int64),
        rewards=np.array(rews, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def as_json_rows_from_dataclasses(items: list[object]) -> list[dict]:
    return [asdict(x) for x in items]
