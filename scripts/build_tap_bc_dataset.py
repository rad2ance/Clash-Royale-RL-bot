from __future__ import annotations

import argparse
import json
from pathlib import Path

from crbot.data import save_episode
from crbot.recording import UiLayout, build_episode_from_logs, load_frame_records, load_tap_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BC episodes from recorded emulator sessions.")
    parser.add_argument("--recordings-dir", type=str, default="recordings")
    parser.add_argument("--out", type=str, default="data/il_tap")
    parser.add_argument("--grid-w", type=int, default=8)
    parser.add_argument("--grid-h", type=int, default=14)
    parser.add_argument("--hand-size", type=int, default=4)
    parser.add_argument("--resize-w", type=int, default=96)
    parser.add_argument("--resize-h", type=int, default=54)
    args = parser.parse_args()

    rec_root = Path(args.recordings_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted([p for p in rec_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No session directories found in: {rec_root}")

    layout = UiLayout(hand_slots=args.hand_size)
    written = 0
    skipped = 0

    for session in sessions:
        frames_file = session / "frames.jsonl"
        taps_file = session / "taps.jsonl"
        metadata_file = session / "metadata.json"
        if not frames_file.exists() or not taps_file.exists() or not metadata_file.exists():
            skipped += 1
            continue

        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        screen_width = int(metadata["screen_width"])
        screen_height = int(metadata["screen_height"])

        frames = load_frame_records(frames_file)
        taps = load_tap_events(taps_file)
        episode = build_episode_from_logs(
            frames=frames,
            taps=taps,
            screen_width=screen_width,
            screen_height=screen_height,
            layout=layout,
            hand_size=args.hand_size,
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            resize_w=args.resize_w,
            resize_h=args.resize_h,
        )
        if episode is None:
            skipped += 1
            continue
        save_episode(out_dir / f"{session.name}.npz", episode)
        written += 1
        print(f"[build] wrote {session.name}.npz with {episode.observations.shape[0]} transitions")

    print(f"[done] dataset episodes={written} skipped={skipped} -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()

