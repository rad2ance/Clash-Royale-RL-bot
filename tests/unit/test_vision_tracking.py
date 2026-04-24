from crbot.vision import (
    DetectedEntity,
    SimpleVisionTracker,
    VisionState,
    track_vision_states,
)


def _state(ts: float, x: float, y: float, *, team: str = "own", score: float = 0.9) -> VisionState:
    return VisionState(
        timestamp=ts,
        width=1280,
        height=720,
        entities=(
            DetectedEntity(
                kind="unit",
                team=team,
                x_norm=x,
                y_norm=y,
                w_norm=0.03,
                h_norm=0.04,
                score=score,
            ),
        ),
    )


def test_tracker_keeps_same_id_for_small_motion() -> None:
    states = [_state(0.0, 0.20, 0.30), _state(0.1, 0.22, 0.31), _state(0.2, 0.24, 0.33)]
    tracked = track_vision_states(states, tracker=SimpleVisionTracker(max_center_distance=0.10))
    ids = [int(s.entities[0].track_id) for s in tracked]
    assert ids == [1, 1, 1]


def test_tracker_creates_new_id_for_large_jump() -> None:
    states = [_state(0.0, 0.10, 0.20), _state(0.1, 0.70, 0.80)]
    tracked = track_vision_states(states, tracker=SimpleVisionTracker(max_center_distance=0.05))
    ids = [int(s.entities[0].track_id) for s in tracked]
    assert ids == [1, 2]


def test_tracker_team_mismatch_does_not_reuse_id() -> None:
    states = [_state(0.0, 0.30, 0.40, team="own"), _state(0.1, 0.31, 0.41, team="enemy")]
    tracked = track_vision_states(states, tracker=SimpleVisionTracker(max_center_distance=0.10))
    ids = [int(s.entities[0].track_id) for s in tracked]
    assert ids == [1, 2]


def test_tracker_survives_short_gap_within_max_missed() -> None:
    tracker = SimpleVisionTracker(max_center_distance=0.10, max_missed_frames=2)
    s0 = _state(0.0, 0.20, 0.30)
    s1 = VisionState(timestamp=0.1, width=1280, height=720, entities=())
    s2 = _state(0.2, 0.23, 0.31)
    out0 = tracker.update(s0)
    out1 = tracker.update(s1)
    out2 = tracker.update(s2)
    assert int(out0.entities[0].track_id) == 1
    # Empty frame emits no visible entities.
    assert len(out1.entities) == 0
    assert int(out2.entities[0].track_id) == 1
