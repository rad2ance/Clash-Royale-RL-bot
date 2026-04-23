from __future__ import annotations

import argparse
from pathlib import Path

from crbot.data import save_episode
from crbot.recording import build_episode_from_frame_actions, load_action_labels, load_frame_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BC episodes from frame+action-label sessions (e.g. replay videos / Clash Royale TV labels)."
    )
    parser.add_argument("--sessions-dir", type=str, default="video_sessions")
    parser.add_argument("--out", type=str, default="data/il_video")
    parser.add_argument("--grid-w", type=int, default=9)
    parser.add_argument("--grid-h", type=int, default=15)
    parser.add_argument("--hand-size", type=int, default=4)
    parser.add_argument("--resize-w", type=int, default=96)
    parser.add_argument("--resize-h", type=int, default=54)
    args = parser.parse_args()

    sess_root = Path(args.sessions_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted([p for p in sess_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No session directories found in: {sess_root}")

    written = 0
    skipped = 0
    for session in sessions:
        frames_file = session / "frames.jsonl"
        labels_file = session / "actions.jsonl"
        if not frames_file.exists() or not labels_file.exists():
            skipped += 1
            continue

        frames = load_frame_records(frames_file)
        labels = load_action_labels(labels_file)
        episode = build_episode_from_frame_actions(
            frames=frames,
            labels=labels,
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

    print(f"[done] episodes={written} skipped={skipped} -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
