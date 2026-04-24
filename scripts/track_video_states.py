from __future__ import annotations

import argparse
from pathlib import Path

from crbot.vision import (
    SimpleVisionTracker,
    load_vision_states_jsonl,
    save_tracked_vision_states_jsonl,
    track_vision_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign stable track IDs to extracted vision states.")
    parser.add_argument("--in", dest="input_path", type=str, required=True, help="Input states JSONL from extract_video_states.py")
    parser.add_argument("--out", type=str, required=True, help="Output tracked states JSONL")
    parser.add_argument("--max-center-distance", type=float, default=0.12)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    states = load_vision_states_jsonl(args.input_path)
    tracker = SimpleVisionTracker(
        max_center_distance=args.max_center_distance,
        max_missed_frames=args.max_missed_frames,
        min_score=args.min_score,
    )
    tracked = track_vision_states(states, tracker=tracker)
    save_tracked_vision_states_jsonl(args.out, tracked)
    print(f"[done] frames={len(tracked)} -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
