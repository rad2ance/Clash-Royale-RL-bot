from pathlib import Path

from PIL import Image

from crbot.recording import (
    ActionLabel,
    FrameRecord,
    TapEvent,
    TouchTracker,
    UiLayout,
    build_episode_from_frame_actions,
    build_episode_from_logs,
    parse_getevent_line,
)


def test_parse_getevent_line_emits_tap_on_release() -> None:
    tracker = TouchTracker()
    lines = [
        "[  100.010000] /dev/input/event6: EV_ABS ABS_MT_POSITION_X 00008000",
        "[  100.020000] /dev/input/event6: EV_ABS ABS_MT_POSITION_Y 00004000",
        "[  100.030000] /dev/input/event6: EV_KEY BTN_TOUCH DOWN",
        "[  100.100000] /dev/input/event6: EV_KEY BTN_TOUCH UP",
    ]
    tap = None
    for line in lines:
        out = parse_getevent_line(
            line=line,
            tracker=tracker,
            screen_width=1000,
            screen_height=2000,
            max_x=65535,
            max_y=65535,
        )
        if out is not None:
            tap = out

    assert tap is not None
    assert abs(tap.screen_x - 500) <= 5
    assert abs(tap.screen_y - 500) <= 5


def test_build_episode_from_logs_pairs_slot_and_arena_taps(tmp_path: Path) -> None:
    frame_1 = tmp_path / "frame_1.png"
    frame_2 = tmp_path / "frame_2.png"
    Image.new("RGB", (1000, 2000), color=(20, 20, 20)).save(frame_1)
    Image.new("RGB", (1000, 2000), color=(80, 80, 80)).save(frame_2)

    frames = [
        FrameRecord(timestamp=10.0, frame_path=str(frame_1)),
        FrameRecord(timestamp=11.0, frame_path=str(frame_2)),
    ]
    taps = [
        TapEvent(timestamp=10.2, raw_x=0, raw_y=0, screen_x=120, screen_y=1820),  # slot 0
        TapEvent(timestamp=10.5, raw_x=0, raw_y=0, screen_x=500, screen_y=900),  # arena placement
    ]

    episode = build_episode_from_logs(
        frames=frames,
        taps=taps,
        screen_width=1000,
        screen_height=2000,
        layout=UiLayout(),
        hand_size=4,
        grid_w=8,
        grid_h=14,
        resize_w=32,
        resize_h=18,
    )
    assert episode is not None
    assert episode.observations.shape == (1, 32 * 18)
    assert int(episode.actions[0]) > 0
    assert bool(episode.dones[-1]) is True


def test_build_episode_from_frame_actions_supports_direct_action_labels(tmp_path: Path) -> None:
    frame_1 = tmp_path / "frame_1.png"
    frame_2 = tmp_path / "frame_2.png"
    Image.new("RGB", (1000, 2000), color=(20, 20, 20)).save(frame_1)
    Image.new("RGB", (1000, 2000), color=(80, 80, 80)).save(frame_2)

    frames = [
        FrameRecord(timestamp=10.0, frame_path=str(frame_1)),
        FrameRecord(timestamp=11.0, frame_path=str(frame_2)),
    ]
    labels = [
        ActionLabel(timestamp=10.2, action=7),
        ActionLabel(timestamp=10.8, action=21),
    ]
    episode = build_episode_from_frame_actions(
        frames=frames,
        labels=labels,
        hand_size=4,
        grid_w=8,
        grid_h=14,
        resize_w=32,
        resize_h=18,
    )
    assert episode is not None
    assert episode.observations.shape == (2, 32 * 18)
    assert episode.actions.tolist() == [7, 21]
    assert bool(episode.dones[-1]) is True
