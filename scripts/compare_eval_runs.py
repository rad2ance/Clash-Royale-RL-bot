from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_eval(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    return {
        "file": str(path),
        "policy": str(data.get("policy", "")),
        "checkpoint": str(data.get("checkpoint", "")),
        "mean_reward": float(summary.get("mean_reward", 0.0)),
        "win_rate": float(summary.get("win_rate", 0.0)),
        "illegal_action_rate": float(summary.get("illegal_action_rate", 0.0)),
        "mean_episode_len": float(summary.get("mean_episode_len", 0.0)),
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("[done] no eval files found.")
        return
    header = (
        f"{'rank':>4}  {'policy':<10}  {'mean_reward':>12}  {'win_rate':>8}  "
        f"{'illegal_rate':>12}  {'mean_len':>9}  file"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>4}  {row['policy']:<10}  {row['mean_reward']:>12.3f}  {row['win_rate']:>8.3f}  "
            f"{row['illegal_action_rate']:>12.4f}  {row['mean_episode_len']:>9.1f}  {row['file']}"
        )


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "policy", "checkpoint", "mean_reward", "win_rate", "illegal_action_rate", "mean_episode_len"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy evaluation JSON outputs.")
    parser.add_argument("--glob", type=str, default="data/eval/*.json", help="Glob for evaluation JSON files.")
    parser.add_argument(
        "--sort-by",
        choices=["mean_reward", "win_rate", "illegal_action_rate", "mean_episode_len"],
        default="mean_reward",
    )
    parser.add_argument("--descending", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--csv-out", type=str, default="", help="Optional CSV export path.")
    args = parser.parse_args()

    files = sorted(Path().glob(args.glob))
    rows = [load_eval(path) for path in files]
    rows.sort(key=lambda r: r[args.sort_by], reverse=bool(args.descending))
    print_table(rows)

    if args.csv_out:
        out = Path(args.csv_out)
        save_csv(out, rows)
        print(f"[done] saved csv: {out.resolve()}")


if __name__ == "__main__":
    main()
