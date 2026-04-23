from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple IL dataset directories into one.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input directories containing .npz episodes.")
    parser.add_argument("--out", type=str, default="data/il_merged")
    parser.add_argument("--prefix-with-source", action="store_true", help="Prefix output filenames with source dir name.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for raw in args.inputs:
        src = Path(raw)
        if not src.exists() or not src.is_dir():
            continue
        source_prefix = src.name
        for p in sorted(src.glob("*.npz")):
            name = p.name
            if args.prefix_with_source:
                name = f"{source_prefix}__{name}"
            dst = out_dir / name
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                i = 1
                while True:
                    alt = out_dir / f"{stem}_{i:03d}{suffix}"
                    if not alt.exists():
                        dst = alt
                        break
                    i += 1
            shutil.copy2(p, dst)
            copied += 1
    print(f"[done] copied={copied} episodes -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
