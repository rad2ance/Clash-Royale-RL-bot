from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from crbot.vision import BaselineCvStateExtractor, save_vision_states_jsonl


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required. Install: pip install -e '.[vision]'") from exc
    return cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract baseline CV entities from replay/video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--out", type=str, default="data/video_states/states.jsonl", help="Output JSONL path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit processed frames (0 means all).")
    parser.add_argument("--stride", type=int, default=3, help="Process every Nth frame.")
    parser.add_argument("--min-blob-area", type=int, default=24)
    parser.add_argument(
        "--min-frame-confidence",
        type=float,
        default=0.10,
        help="Minimum confidence required before entity extraction is emitted.",
    )
    parser.add_argument(
        "--skip-low-confidence",
        action="store_true",
        help="Drop low-confidence frames from JSONL output entirely.",
    )
    args = parser.parse_args()

    cv2 = _require_cv2()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    extractor = BaselineCvStateExtractor(
        min_blob_area=args.min_blob_area,
        min_frame_confidence_for_entities=args.min_frame_confidence,
    )
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    states = []
    frame_idx = 0
    processed = 0
    skipped_low_conf = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.stride > 1 and (frame_idx % args.stride) != 0:
                frame_idx += 1
                continue
            ts = frame_idx / max(1e-6, fps)
            state = extractor.extract(frame, ts)
            state = replace(state, source_frame_index=int(frame_idx))
            if args.skip_low_conf and state.frame_confidence < args.min_frame_confidence:
                skipped_low_conf += 1
                frame_idx += 1
                continue
            states.append(state)
            processed += 1
            frame_idx += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        cap.release()

    save_vision_states_jsonl(args.out, states)
    print(
        f"[done] states={len(states)} skipped_low_conf={skipped_low_conf} min_conf={args.min_frame_confidence:.2f}"
        f" -> {Path(args.out).resolve()}"
    )


if __name__ == "__main__":
    main()
