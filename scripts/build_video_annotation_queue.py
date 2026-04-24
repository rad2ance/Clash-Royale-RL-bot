from __future__ import annotations

import argparse
import json
from pathlib import Path

from crbot.vision import (
    build_annotation_queue,
    load_state_rows_jsonl,
    save_annotation_queue_jsonl,
    summarize_annotation_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prioritized frame queue for manual video annotation."
    )
    parser.add_argument("--states", type=str, required=True, help="Input states/tracks JSONL.")
    parser.add_argument("--out-queue", type=str, default="data/video_states/annotation_queue.jsonl")
    parser.add_argument("--out-summary", type=str, default="data/video_states/annotation_summary.json")
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--low-conf-threshold", type=float, default=0.35)
    parser.add_argument("--low-entity-score-threshold", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_state_rows_jsonl(args.states)
    queue = build_annotation_queue(
        rows,
        top_k=args.top_k,
        low_conf_threshold=args.low_conf_threshold,
        low_entity_score_threshold=args.low_entity_score_threshold,
    )
    summary = summarize_annotation_budget(rows)
    summary["queue_size"] = int(len(queue))
    summary["thresholds"] = {
        "low_conf_threshold": float(args.low_conf_threshold),
        "low_entity_score_threshold": float(args.low_entity_score_threshold),
    }
    summary["top_queue_preview"] = [
        {
            "frame_index": int(x.frame_index),
            "source_frame_index": int(x.source_frame_index),
            "timestamp": float(x.timestamp),
            "priority": float(x.priority),
            "reason_flags": list(x.reason_flags),
        }
        for x in queue[:10]
    ]

    save_annotation_queue_jsonl(args.out_queue, queue)
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"[done] frames={summary['n_frames']} queue={len(queue)} -> "
        f"{Path(args.out_queue).resolve()} | {out_summary.resolve()}"
    )


if __name__ == "__main__":
    main()
