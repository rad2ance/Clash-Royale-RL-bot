from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def assign_split(keys: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Ratios must satisfy: train>0, val>=0, train+val<1.")
    rng = random.Random(seed)
    keys = sorted(set(keys))
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    out: dict[str, str] = {}
    for k in keys[:n_train]:
        out[k] = "train"
    for k in keys[n_train : n_train + n_val]:
        out[k] = "val"
    for k in keys[n_train + n_val :]:
        out[k] = "test"
    return out


def group_key(path: Path, by_source: bool) -> str:
    name = path.stem
    if by_source and "__" in name:
        return name.split("__", 1)[0]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Split IL episodes into train/val/test directories.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .npz episodes.")
    parser.add_argument("--out", type=str, default="data/il_split")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="Split by source prefix (filename prefix before '__') to reduce cross-source leakage.",
    )
    args = parser.parse_args()

    src = Path(args.data_dir)
    files = sorted(src.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz episodes found in: {src}")

    out_root = Path(args.out)
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    keys = [group_key(p, args.group_by_source) for p in files]
    key_to_split = assign_split(keys, args.train_ratio, args.val_ratio, args.seed)

    manifest: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for p in files:
        k = group_key(p, args.group_by_source)
        split = key_to_split[k]
        dst = out_root / split / p.name
        shutil.copy2(p, dst)
        manifest.append({"episode": p.name, "group": k, "split": split})
        counts[split] += 1

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] split complete -> {out_root.resolve()}")
    print(f"[done] counts: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[done] manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
