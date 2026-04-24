from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from crbot.cards import build_registry_review_backlog, default_registry_path, summarize_registry_review_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prioritized backlog for card registry review.")
    parser.add_argument("--registry", type=str, default=str(default_registry_path()))
    parser.add_argument("--out-jsonl", type=str, default="data/cards/review_backlog.jsonl")
    parser.add_argument("--out-summary", type=str, default="data/cards/review_backlog_summary.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Include all cards, not only cards tagged needs_review.",
    )
    parser.add_argument(
        "--no-fallback-include-reviewed",
        dest="fallback_include_reviewed",
        action="store_false",
        default=True,
        help="Disable auto-fallback to include reviewed cards when needs_review backlog is empty.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    only_needs_review = not args.include_reviewed
    backlog = build_registry_review_backlog(
        raw,
        limit=args.limit,
        only_needs_review=only_needs_review,
    )
    fallback_used = False
    if not backlog and only_needs_review and args.fallback_include_reviewed:
        backlog = build_registry_review_backlog(
            raw,
            limit=args.limit,
            only_needs_review=False,
        )
        fallback_used = True

    summary = summarize_registry_review_state(raw)
    summary["backlog_size"] = int(len(backlog))
    summary["include_reviewed"] = bool(args.include_reviewed)
    summary["fallback_include_reviewed_used"] = bool(fallback_used)
    summary["top_preview"] = backlog[:10]

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in backlog:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if fallback_used:
        print("[note] no needs_review cards found; falling back to include reviewed cards.")
    print(f"[done] backlog={len(backlog)} -> {out_jsonl.resolve()} | {out_summary.resolve()}")


if __name__ == "__main__":
    main()
