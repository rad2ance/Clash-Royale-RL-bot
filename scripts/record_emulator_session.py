from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from crbot.emulator import AdbClient, BlueStacksInstance
from crbot.recording import FrameRecord, TapEvent, TouchTracker, parse_getevent_line, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a BlueStacks session with frames and touch events from adb getevent."
    )
    parser.add_argument("--out", type=str, default="recordings")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--adb-path", type=str, default="adb")
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--input-device", type=str, default="")
    parser.add_argument("--raw-max-x", type=int, default=0)
    parser.add_argument("--raw-max-y", type=int, default=0)
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"session_{timestamp}"
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    adb = AdbClient(adb_path=args.adb_path)
    bs = BlueStacksInstance(adb=adb, host=args.host, port=args.port)
    bs.connect()

    first_frame_path = frame_dir / "frame_000000.png"
    bs.take_screenshot(first_frame_path)
    from PIL import Image

    with Image.open(first_frame_path) as im:
        screen_width, screen_height = im.size

    input_device = args.input_device.strip() or adb.find_touch_input_device()
    if not input_device:
        raise RuntimeError("Could not auto-detect touch input device. Pass --input-device explicitly.")

    max_x = args.raw_max_x
    max_y = args.raw_max_y
    if max_x <= 0 or max_y <= 0:
        ax, ay = adb.get_touch_axis_max(input_device)
        max_x = max_x if max_x > 0 else ax
        max_y = max_y if max_y > 0 else ay

    proc = adb.shell_stream("getevent", "-lt", input_device)
    if proc.stdout is None:
        raise RuntimeError("Failed to open adb getevent stdout stream.")

    tracker = TouchTracker()
    taps: list[TapEvent] = []
    tap_lock = threading.Lock()
    stop_event = threading.Event()

    def read_touch_stream() -> None:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            tap = parse_getevent_line(
                line=line,
                tracker=tracker,
                screen_width=screen_width,
                screen_height=screen_height,
                max_x=max_x,
                max_y=max_y,
            )
            if tap is not None:
                with tap_lock:
                    taps.append(tap)

    reader = threading.Thread(target=read_touch_stream, daemon=True)
    reader.start()

    start = time.time()
    next_capture = start
    frame_records: list[FrameRecord] = [FrameRecord(timestamp=start, frame_path=str(first_frame_path))]
    frame_idx = 1

    try:
        while True:
            now = time.time()
            if now - start >= args.duration:
                break
            if now >= next_capture:
                frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
                bs.take_screenshot(frame_path)
                frame_records.append(FrameRecord(timestamp=now, frame_path=str(frame_path)))
                frame_idx += 1
                next_capture = now + 1.0 / max(args.fps, 0.1)
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("[record] interrupted by user, finalizing...")
    finally:
        stop_event.set()
        proc.terminate()
        reader.join(timeout=1.0)

    with tap_lock:
        taps_snapshot = list(taps)

    save_jsonl(run_dir / "frames.jsonl", [asdict(x) for x in frame_records])
    save_jsonl(run_dir / "taps.jsonl", [asdict(x) for x in taps_snapshot])
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "host": args.host,
                "port": args.port,
                "input_device": input_device,
                "screen_width": screen_width,
                "screen_height": screen_height,
                "raw_max_x": max_x,
                "raw_max_y": max_y,
                "duration_s": args.duration,
                "fps": args.fps,
            },
            f,
            indent=2,
        )
    print(f"[done] session saved: {run_dir.resolve()}")
    print(f"[done] frames={len(frame_records)} taps={len(taps_snapshot)}")


if __name__ == "__main__":
    main()
