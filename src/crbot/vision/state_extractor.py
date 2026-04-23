from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


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
    ) -> None:
        self.min_blob_area = int(min_blob_area)
        self.blur_ksize = int(max(1, blur_ksize))

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

    def _extract_team_entities(self, frame_bgr: np.ndarray, hsv: np.ndarray, team: str, cv2) -> list[DetectedEntity]:
        h, w = frame_bgr.shape[:2]
        mask = self._mask_team(hsv, team, cv2)
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
        k = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        entities: list[DetectedEntity] = []
        entities.extend(self._extract_team_entities(frame_bgr, hsv, team="own", cv2=cv2))
        entities.extend(self._extract_team_entities(frame_bgr, hsv, team="enemy", cv2=cv2))
        return VisionState(
            timestamp=float(timestamp),
            width=int(w),
            height=int(h),
            entities=tuple(entities),
        )
