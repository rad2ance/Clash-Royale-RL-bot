from pathlib import Path

from crbot.vision import DetectedEntity, VisionState, save_vision_states_jsonl


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
