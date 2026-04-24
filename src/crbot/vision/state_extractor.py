from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class UiAnchors:
    """
    Normalized UI anchors used to scope arena detections and diagnostics.
    """

    arena_left: float
    arena_right: float
    arena_top: float
    arena_bottom: float
    hand_left: float
    hand_right: float
    hand_top: float
    hand_bottom: float


@dataclass(frozen=True)
class DetectedEntity:
    kind: str  # "unit" | "tower" | "spell_effect" | ...
    team: str  # "own" | "enemy" | "neutral"
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float
    score: float


@dataclass(frozen=True)
class VisionState:
    timestamp: float
    width: int
    height: int
    entities: tuple[DetectedEntity, ...]
    ui_anchors: UiAnchors | None = None
    frame_confidence: float = 1.0


class StateExtractor(Protocol):
    def extract(self, frame_bgr: np.ndarray, timestamp: float) -> VisionState:
        ...


def save_vision_states_jsonl(path: str | Path, states: list[VisionState]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in states:
            row = asdict(s)
            row["entities"] = [asdict(e) for e in s.entities]
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_vision_states_jsonl(path: str | Path) -> list[VisionState]:
    p = Path(path)
    states: list[VisionState] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            anchors_raw = row.get("ui_anchors")
            anchors = None
            if isinstance(anchors_raw, dict):
                anchors = UiAnchors(
                    arena_left=float(anchors_raw["arena_left"]),
                    arena_right=float(anchors_raw["arena_right"]),
                    arena_top=float(anchors_raw["arena_top"]),
                    arena_bottom=float(anchors_raw["arena_bottom"]),
                    hand_left=float(anchors_raw["hand_left"]),
                    hand_right=float(anchors_raw["hand_right"]),
                    hand_top=float(anchors_raw["hand_top"]),
                    hand_bottom=float(anchors_raw["hand_bottom"]),
                )
            entities = tuple(
                DetectedEntity(
                    kind=str(e["kind"]),
                    team=str(e["team"]),
                    x_norm=float(e["x_norm"]),
                    y_norm=float(e["y_norm"]),
                    w_norm=float(e["w_norm"]),
                    h_norm=float(e["h_norm"]),
                    score=float(e.get("score", 0.0)),
                )
                for e in row.get("entities", [])
            )
            states.append(
                VisionState(
                    timestamp=float(row["timestamp"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    entities=entities,
                    ui_anchors=anchors,
                    frame_confidence=float(row.get("frame_confidence", 1.0)),
                )
            )
    return states


def default_ui_anchors(width: int, height: int) -> UiAnchors:
    """
    Return baseline normalized UI anchors for a Clash Royale portrait layout.
    """

    # The constants are tuned for portrait sessions and intentionally conservative.
    # We slightly tighten the arena in landscape captures to reduce UI leakage.
    if height >= width:
        arena_top = 0.08
        arena_bottom = 0.79
        hand_top = 0.80
        hand_bottom = 0.98
    else:
        arena_top = 0.06
        arena_bottom = 0.76
        hand_top = 0.78
        hand_bottom = 0.98
    return UiAnchors(
        arena_left=0.02,
        arena_right=0.98,
        arena_top=arena_top,
        arena_bottom=arena_bottom,
        hand_left=0.03,
        hand_right=0.97,
        hand_top=hand_top,
        hand_bottom=hand_bottom,
    )


def estimate_frame_confidence(frame_bgr: np.ndarray, anchors: UiAnchors | None = None) -> float:
    """
    Estimate frame quality/confidence using brightness + texture in the arena ROI.

    This is a lightweight heuristic for filtering extremely dark/blank/noisy frames.
    """

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return 0.0
    h, w = frame_bgr.shape[:2]
    if h <= 1 or w <= 1:
        return 0.0
    ui = anchors or default_ui_anchors(width=w, height=h)
    x0 = int(np.clip(np.floor(ui.arena_left * w), 0, w - 1))
    x1 = int(np.clip(np.ceil(ui.arena_right * w), x0 + 1, w))
    y0 = int(np.clip(np.floor(ui.arena_top * h), 0, h - 1))
    y1 = int(np.clip(np.ceil(ui.arena_bottom * h), y0 + 1, h))
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    roi_f = roi.astype(np.float32) / 255.0
    gray = roi_f.mean(axis=2)
    brightness = float(gray.mean())
    texture = float(gray.std())
    channel_spread = float(roi_f.mean(axis=(0, 1)).std())

    brightness_score = float(np.clip((brightness - 0.08) / 0.45, 0.0, 1.0))
    texture_score = float(np.clip(texture / 0.20, 0.0, 1.0))
    color_score = float(np.clip(channel_spread / 0.10, 0.0, 1.0))
    score = 0.45 * brightness_score + 0.45 * texture_score + 0.10 * color_score
    return float(np.clip(score, 0.0, 1.0))


def _norm_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[float, float, float, float]:
    cx = (x + 0.5 * w) / max(1, width)
    cy = (y + 0.5 * h) / max(1, height)
    nw = w / max(1, width)
    nh = h / max(1, height)
    return float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0)), float(np.clip(nw, 0.0, 1.0)), float(
        np.clip(nh, 0.0, 1.0)
    )


class BaselineCvStateExtractor:
    """
    Baseline heuristic extractor for replay/video bootstrapping.

    It uses simple HSV color segmentation for team-colored blobs and contour
    filtering. This is intentionally a bootstrap baseline, not production CV.
    """

    def __init__(
        self,
        min_blob_area: int = 24,
        blur_ksize: int = 3,
        min_frame_confidence_for_entities: float = 0.10,
    ) -> None:
        self.min_blob_area = int(min_blob_area)
        self.blur_ksize = int(max(1, blur_ksize))
        self.min_frame_confidence_for_entities = float(np.clip(min_frame_confidence_for_entities, 0.0, 1.0))

    def _require_cv2(self):
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("opencv-python is required for vision extraction. Install: pip install -e '.[vision]'") from exc
        return cv2

    def _mask_team(self, hsv: np.ndarray, team: str, cv2):
        if team == "own":
            # Blue-ish
            lower = np.array([90, 40, 40], dtype=np.uint8)
            upper = np.array([135, 255, 255], dtype=np.uint8)
        else:
            # Red-ish (wrap around hue)
            lower1 = np.array([0, 45, 45], dtype=np.uint8)
            upper1 = np.array([12, 255, 255], dtype=np.uint8)
            lower2 = np.array([165, 45, 45], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        return cv2.inRange(hsv, lower, upper)

    def _extract_team_entities(
        self,
        frame_bgr: np.ndarray,
        hsv: np.ndarray,
        team: str,
        anchors: UiAnchors,
        cv2,
    ) -> list[DetectedEntity]:
        h, w = frame_bgr.shape[:2]
        mask = self._mask_team(hsv, team, cv2)
        arena_mask = np.zeros((h, w), dtype=np.uint8)
        x0 = int(np.clip(np.floor(anchors.arena_left * w), 0, w - 1))
        x1 = int(np.clip(np.ceil(anchors.arena_right * w), x0 + 1, w))
        y0 = int(np.clip(np.floor(anchors.arena_top * h), 0, h - 1))
        y1 = int(np.clip(np.ceil(anchors.arena_bottom * h), y0 + 1, h))
        arena_mask[y0:y1, x0:x1] = 255
        mask = cv2.bitwise_and(mask, arena_mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[DetectedEntity] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_blob_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x_norm, y_norm, w_norm, h_norm = _norm_box(x, y, bw, bh, w, h)
            score = float(np.clip(area / float(w * h), 0.0, 1.0))
            out.append(
                DetectedEntity(
                    kind="unit",
                    team=team,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    w_norm=w_norm,
                    h_norm=h_norm,
                    score=score,
                )
            )
        return out

    def extract(self, frame_bgr: np.ndarray, timestamp: float) -> VisionState:
        cv2 = self._require_cv2()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape [H, W, 3]")
        h, w = frame_bgr.shape[:2]
        anchors = default_ui_anchors(width=w, height=h)
        frame_confidence = estimate_frame_confidence(frame_bgr, anchors=anchors)
        k = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        entities: list[DetectedEntity] = []
        if frame_confidence >= self.min_frame_confidence_for_entities:
            entities.extend(self._extract_team_entities(frame_bgr, hsv, team="own", anchors=anchors, cv2=cv2))
            entities.extend(self._extract_team_entities(frame_bgr, hsv, team="enemy", anchors=anchors, cv2=cv2))
        return VisionState(
            timestamp=float(timestamp),
            width=int(w),
            height=int(h),
            entities=tuple(entities),
            ui_anchors=anchors,
            frame_confidence=float(frame_confidence),
        )
