from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnnotationCandidate:
    frame_index: int
    timestamp: float
    priority: float
    reason_flags: tuple[str, ...]
    frame_confidence: float
    entity_count: int
    mean_entity_score: float
    low_score_ratio: float
    new_tracks: int
    motion_score: float


def load_state_rows_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _entity_center(entity: dict) -> tuple[float, float]:
    return float(entity.get("x_norm", 0.0)), float(entity.get("y_norm", 0.0))


def build_annotation_queue(
    rows: list[dict],
    *,
    top_k: int = 300,
    low_conf_threshold: float = 0.35,
    low_entity_score_threshold: float = 0.25,
) -> list[AnnotationCandidate]:
    candidates: list[AnnotationCandidate] = []
    prev_centers_by_track: dict[int, tuple[float, float]] = {}
    seen_track_ids: set[int] = set()

    for idx, row in enumerate(rows):
        timestamp = float(row.get("timestamp", float(idx)))
        frame_conf = float(row.get("frame_confidence", 1.0))
        entities = list(row.get("entities", []))
        entity_count = int(len(entities))

        scores = [float(e.get("score", 0.0)) for e in entities]
        mean_entity_score = float(np.mean(scores)) if scores else 0.0
        low_score_ratio = float(np.mean([s < low_entity_score_threshold for s in scores])) if scores else 0.0

        track_ids = [int(e["track_id"]) for e in entities if "track_id" in e]
        new_tracks = sum(1 for tid in track_ids if tid not in seen_track_ids)
        for tid in track_ids:
            seen_track_ids.add(tid)

        motion_vals: list[float] = []
        next_prev_centers = dict(prev_centers_by_track)
        for e in entities:
            if "track_id" not in e:
                continue
            tid = int(e["track_id"])
            cx, cy = _entity_center(e)
            prev = prev_centers_by_track.get(tid)
            if prev is not None:
                dx = cx - prev[0]
                dy = cy - prev[1]
                motion_vals.append(float(np.sqrt(dx * dx + dy * dy)))
            next_prev_centers[tid] = (cx, cy)
        prev_centers_by_track = next_prev_centers
        motion_score = float(np.clip(np.mean(motion_vals) if motion_vals else 0.0, 0.0, 1.0))

        uncertainty = (
            0.45 * float(np.clip(1.0 - frame_conf, 0.0, 1.0))
            + 0.30 * float(np.clip(low_score_ratio, 0.0, 1.0))
            + 0.15 * float(np.clip(new_tracks / 5.0, 0.0, 1.0))
            + 0.10 * float(np.clip(motion_score / 0.10, 0.0, 1.0))
        )

        flags: list[str] = []
        if frame_conf < low_conf_threshold:
            flags.append("low_frame_confidence")
        if mean_entity_score < low_entity_score_threshold and entity_count > 0:
            flags.append("low_entity_score")
        if new_tracks > 0:
            flags.append("new_tracks")
        if motion_score > 0.04:
            flags.append("high_motion")
        if entity_count == 0:
            flags.append("empty_arena_detection")

        candidates.append(
            AnnotationCandidate(
                frame_index=int(idx),
                timestamp=timestamp,
                priority=float(np.clip(uncertainty, 0.0, 1.0)),
                reason_flags=tuple(flags),
                frame_confidence=frame_conf,
                entity_count=entity_count,
                mean_entity_score=mean_entity_score,
                low_score_ratio=low_score_ratio,
                new_tracks=int(new_tracks),
                motion_score=motion_score,
            )
        )

    ordered = sorted(candidates, key=lambda c: (c.priority, c.entity_count), reverse=True)
    return ordered[: max(0, int(top_k))]


def summarize_annotation_budget(rows: list[dict]) -> dict:
    n_frames = int(len(rows))
    if n_frames <= 0:
        return {
            "n_frames": 0,
            "recommended_labels": {
                "quick_bootstrap": 0,
                "strong_baseline": 0,
                "robust_detector_seed": 0,
            },
        }

    return {
        "n_frames": n_frames,
        "recommended_labels": {
            "quick_bootstrap": int(min(150, max(40, round(0.01 * n_frames)))),
            "strong_baseline": int(min(600, max(150, round(0.03 * n_frames)))),
            "robust_detector_seed": int(min(1500, max(400, round(0.07 * n_frames)))),
        },
    }


def save_annotation_queue_jsonl(path: str | Path, rows: list[AnnotationCandidate]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
