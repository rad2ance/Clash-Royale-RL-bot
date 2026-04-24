from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .state_extractor import DetectedEntity, UiAnchors, VisionState


@dataclass(frozen=True)
class TrackedEntity:
    track_id: int
    kind: str
    team: str
    x_norm: float
    y_norm: float
    w_norm: float
    h_norm: float
    score: float
    age: int
    missed_frames: int


@dataclass(frozen=True)
class TrackedVisionState:
    timestamp: float
    width: int
    height: int
    entities: tuple[TrackedEntity, ...]
    ui_anchors: UiAnchors | None = None
    frame_confidence: float = 1.0
    source_frame_index: int | None = None


@dataclass
class _Track:
    track_id: int
    entity: DetectedEntity
    age: int = 1
    missed_frames: int = 0


class SimpleVisionTracker:
    """
    Greedy multi-object tracker for bootstrap video labeling.

    Matching uses center distance with team/kind consistency and bounded gap.
    """

    def __init__(self, max_center_distance: float = 0.12, max_missed_frames: int = 2, min_score: float = 0.0) -> None:
        self.max_center_distance = float(max(0.0, max_center_distance))
        self.max_missed_frames = int(max(0, max_missed_frames))
        self.min_score = float(min_score)
        self._next_track_id = 1
        self._tracks: list[_Track] = []

    @staticmethod
    def _center_distance(a: DetectedEntity, b: DetectedEntity) -> float:
        dx = float(a.x_norm - b.x_norm)
        dy = float(a.y_norm - b.y_norm)
        return float(np.sqrt(dx * dx + dy * dy))

    def _as_tracked(self, t: _Track) -> TrackedEntity:
        e = t.entity
        return TrackedEntity(
            track_id=int(t.track_id),
            kind=e.kind,
            team=e.team,
            x_norm=float(e.x_norm),
            y_norm=float(e.y_norm),
            w_norm=float(e.w_norm),
            h_norm=float(e.h_norm),
            score=float(e.score),
            age=int(t.age),
            missed_frames=int(t.missed_frames),
        )

    def update(self, state: VisionState) -> TrackedVisionState:
        detections = [e for e in state.entities if float(e.score) >= self.min_score]
        if not detections:
            for tr in self._tracks:
                tr.missed_frames += 1
            self._tracks = [tr for tr in self._tracks if tr.missed_frames <= self.max_missed_frames]
            return TrackedVisionState(
                timestamp=state.timestamp,
                width=state.width,
                height=state.height,
                entities=tuple(),
                ui_anchors=state.ui_anchors,
                frame_confidence=state.frame_confidence,
                source_frame_index=state.source_frame_index,
            )

        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if tr.entity.team != det.team or tr.entity.kind != det.kind:
                    continue
                dist = self._center_distance(tr.entity, det)
                if dist <= self.max_center_distance:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda x: x[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            tr = self._tracks[ti]
            tr.entity = detections[di]
            tr.age += 1
            tr.missed_frames = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, tr in enumerate(self._tracks):
            if ti not in matched_tracks:
                tr.missed_frames += 1
        self._tracks = [tr for tr in self._tracks if tr.missed_frames <= self.max_missed_frames]

        new_tracks: list[_Track] = []
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            new_tr = _Track(
                track_id=self._next_track_id,
                entity=det,
                age=1,
                missed_frames=0,
            )
            self._tracks.append(new_tr)
            new_tracks.append(new_tr)
            self._next_track_id += 1

        visible_tracks = [self._tracks[i] for i in sorted(matched_tracks)]
        visible_tracks.extend(new_tracks)
        tracked = sorted((self._as_tracked(tr) for tr in visible_tracks), key=lambda x: x.track_id)
        return TrackedVisionState(
            timestamp=state.timestamp,
            width=state.width,
            height=state.height,
            entities=tuple(tracked),
            ui_anchors=state.ui_anchors,
            frame_confidence=state.frame_confidence,
            source_frame_index=state.source_frame_index,
        )


def track_vision_states(states: list[VisionState], tracker: SimpleVisionTracker | None = None) -> list[TrackedVisionState]:
    tr = tracker or SimpleVisionTracker()
    out: list[TrackedVisionState] = []
    for state in states:
        out.append(tr.update(state))
    return out


def save_tracked_vision_states_jsonl(path: str | Path, states: list[TrackedVisionState]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in states:
            row = asdict(s)
            row["entities"] = [asdict(e) for e in s.entities]
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
