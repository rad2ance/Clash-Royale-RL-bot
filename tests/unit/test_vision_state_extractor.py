from pathlib import Path

import numpy as np

from crbot.vision import (
    DetectedEntity,
    VisionState,
    default_ui_anchors,
    estimate_frame_confidence,
    load_vision_states_jsonl,
    save_vision_states_jsonl,
)


def test_save_vision_states_jsonl_writes_rows(tmp_path: Path) -> None:
    states = [
        VisionState(
            timestamp=1.0,
            width=640,
            height=360,
            entities=(
                DetectedEntity(
                    kind="unit",
                    team="own",
                    x_norm=0.5,
                    y_norm=0.5,
                    w_norm=0.1,
                    h_norm=0.1,
                    score=0.8,
                ),
            ),
        )
    ]
    out = tmp_path / "states.jsonl"
    save_vision_states_jsonl(out, states)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"timestamp": 1.0' in lines[0]


def test_load_vision_states_jsonl_roundtrip() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "states.jsonl"
        anchors = default_ui_anchors(width=640, height=360)
        states = [
            VisionState(
                timestamp=1.0,
                width=640,
                height=360,
                entities=(
                    DetectedEntity(
                        kind="unit",
                        team="own",
                        x_norm=0.5,
                        y_norm=0.5,
                        w_norm=0.1,
                        h_norm=0.1,
                        score=0.8,
                    ),
                ),
                ui_anchors=anchors,
                frame_confidence=0.75,
            )
        ]
        save_vision_states_jsonl(out, states)
        loaded = load_vision_states_jsonl(out)
        assert len(loaded) == 1
        assert loaded[0].width == 640
        assert loaded[0].ui_anchors is not None
        assert loaded[0].entities[0].team == "own"
        assert abs(float(loaded[0].frame_confidence) - 0.75) < 1e-6


def test_default_ui_anchors_are_normalized() -> None:
    anchors = default_ui_anchors(width=1080, height=1920)
    assert 0.0 <= anchors.arena_left < anchors.arena_right <= 1.0
    assert 0.0 <= anchors.arena_top < anchors.arena_bottom <= 1.0
    assert 0.0 <= anchors.hand_left < anchors.hand_right <= 1.0
    assert 0.0 <= anchors.hand_top < anchors.hand_bottom <= 1.0


def test_estimate_frame_confidence_prefers_textured_bright_arena() -> None:
    anchors = default_ui_anchors(width=240, height=400)
    low = np.zeros((400, 240, 3), dtype=np.uint8)
    high = np.zeros((400, 240, 3), dtype=np.uint8)
    y0 = int(anchors.arena_top * 400)
    y1 = int(anchors.arena_bottom * 400)
    x0 = int(anchors.arena_left * 240)
    x1 = int(anchors.arena_right * 240)

    rng = np.random.default_rng(0)
    high[y0:y1, x0:x1] = rng.integers(80, 220, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)

    low_conf = estimate_frame_confidence(low, anchors=anchors)
    high_conf = estimate_frame_confidence(high, anchors=anchors)
    assert 0.0 <= low_conf <= 1.0
    assert 0.0 <= high_conf <= 1.0
    assert high_conf > low_conf
