from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from crbot.sim import CrLikeSimEnv


def overlay_frame(
    frame: Image.Image,
    step_idx: int,
    action: int | None,
    reward: float | None,
    info: dict | None,
) -> Image.Image:
    out = frame.copy()
    draw = ImageDraw.Draw(out)
    panel_w = 220
    panel_h = 88
    draw.rectangle((6, 6, 6 + panel_w, 6 + panel_h), fill=(0, 0, 0))
    legal = "" if info is None else f" legal={info.get('legal_action')}"
    card_type = "" if info is None else f" card={info.get('card_type')}"
    draw.text((12, 12), f"step={step_idx}", fill=(255, 255, 255))
    draw.text((12, 28), f"action={action if action is not None else '-'}{legal}", fill=(220, 220, 220))
    draw.text((12, 44), f"reward={0.0 if reward is None else reward:.3f}{card_type}", fill=(220, 220, 220))
    draw.text((12, 60), "blue=time  green=elixir", fill=(180, 180, 200))
    draw.text((12, 74), "red square=last action", fill=(255, 164, 164))
    return out


def metric_row(
    step_idx: int,
    action: int | None,
    reward: float | None,
    terminated: bool,
    truncated: bool,
    info: dict | None,
    env: CrLikeSimEnv,
) -> dict[str, object]:
    info = info or {}
    return {
        "step": step_idx,
        "action": -1 if action is None else int(action),
        "reward": 0.0 if reward is None else float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "legal_action": bool(info.get("legal_action", True)),
        "illegal_reason": str(info.get("illegal_reason", "")),
        "card_type": str(info.get("card_type", "")),
        "spent_elixir": float(info.get("spent_elixir", 0.0)),
        "damage_to_enemy": float(info.get("damage_to_enemy", 0.0)),
        "damage_to_self": float(info.get("damage_to_self", 0.0)),
        "env_elixir": float(env.elixir),
        "env_time_left": float(env.time_left),
        "own_king_hp": float(env.own_king_hp),
        "enemy_king_hp": float(env.enemy_king_hp),
    }


def save_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a simulator episode to GIF.")
    parser.add_argument("--steps", type=int, default=180, help="Maximum environment steps to render.")
    parser.add_argument("--fps", type=int, default=8, help="Output GIF frames per second.")
    parser.add_argument("--seed", type=int, default=42, help="Environment RNG seed.")
    parser.add_argument("--allow-illegal-actions", action="store_true", help="Sample from full action space.")
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional CSV output for per-step metrics.",
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay step/action/reward text panel on frames.",
    )
    parser.add_argument("--out", type=str, default="data/sim_viz/episode.gif", help="Output GIF path.")
    args = parser.parse_args()

    env = CrLikeSimEnv(seed=args.seed)
    _, _ = env.reset(seed=args.seed)

    frames: list[Image.Image] = []
    metrics: list[dict[str, object]] = []

    first_frame = Image.fromarray(env.render(), mode="RGB")
    if args.annotate:
        first_frame = overlay_frame(first_frame, step_idx=0, action=None, reward=None, info=None)
    frames.append(first_frame)
    metrics.append(metric_row(step_idx=0, action=None, reward=None, terminated=False, truncated=False, info=None, env=env))

    for step_idx in range(1, args.steps + 1):
        if args.allow_illegal_actions:
            action = int(env.action_space.sample())
        else:
            action = env.sample_legal_action()
        _, reward, terminated, truncated, info = env.step(action)
        frame = Image.fromarray(env.render(), mode="RGB")
        if args.annotate:
            frame = overlay_frame(frame, step_idx=step_idx, action=action, reward=reward, info=info)
        frames.append(frame)
        metrics.append(
            metric_row(
                step_idx=step_idx,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                env=env,
            )
        )
        if terminated or truncated:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / max(args.fps, 1))))
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"[done] saved visualization: {out.resolve()} ({len(frames)} frames)")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        save_metrics_csv(metrics_path, metrics)
        print(f"[done] saved metrics: {metrics_path.resolve()} ({len(metrics)} rows)")


if __name__ == "__main__":
    main()
