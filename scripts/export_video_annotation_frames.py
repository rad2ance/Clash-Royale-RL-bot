from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required. Install: pip install -e '.[vision]'") from exc
    return cv2


def _load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-priority annotation frames from a source video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--queue", type=str, required=True, help="Annotation queue JSONL from build_video_annotation_queue.py")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--min-priority", type=float, default=0.0)
    args = parser.parse_args()

    cv2 = _require_cv2()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    queue_rows = _load_jsonl(args.queue)
    queue_rows = [r for r in queue_rows if float(r.get("priority", 0.0)) >= float(args.min_priority)]
    queue_rows = sorted(queue_rows, key=lambda r: float(r.get("priority", 0.0)), reverse=True)[: max(0, int(args.top_k))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    written = 0
    seen_frame_indices: set[int] = set()
    try:
        with manifest_path.open("w", encoding="utf-8") as mf:
            for rank, row in enumerate(queue_rows, start=1):
                fi = int(row.get("source_frame_index", row.get("frame_index", -1)))
                if fi < 0 or fi in seen_frame_indices:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
                ok, frame = cap.read()
                if not ok:
                    continue
                seen_frame_indices.add(fi)
                ts = float(row.get("timestamp", 0.0))
                pr = float(row.get("priority", 0.0))
                out_name = f"{rank:04d}_f{fi:06d}_t{ts:09.3f}_p{pr:0.3f}.png"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), frame)
                record = {
                    "rank": int(rank),
                    "frame_index": fi,
                    "timestamp": ts,
                    "priority": pr,
                    "reason_flags": row.get("reason_flags", []),
                    "path": out_name,
                }
                mf.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1
    finally:
        cap.release()

    print(f"[done] exported={written} -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
