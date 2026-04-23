from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from crbot.sim import CrLikeSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a simulator episode to GIF.")
    parser.add_argument("--steps", type=int, default=180, help="Maximum environment steps to render.")
    parser.add_argument("--fps", type=int, default=8, help="Output GIF frames per second.")
    parser.add_argument("--seed", type=int, default=42, help="Environment RNG seed.")
    parser.add_argument("--allow-illegal-actions", action="store_true", help="Sample from full action space.")
    parser.add_argument("--out", type=str, default="data/sim_viz/episode.gif", help="Output GIF path.")
    args = parser.parse_args()

    env = CrLikeSimEnv(seed=args.seed)
    _, _ = env.reset(seed=args.seed)

    frames: list[Image.Image] = [Image.fromarray(env.render(), mode="RGB")]
    for _ in range(args.steps):
        if args.allow_illegal_actions:
            action = int(env.action_space.sample())
        else:
            action = env.sample_legal_action()
        _, _, terminated, truncated, _ = env.step(action)
        frames.append(Image.fromarray(env.render(), mode="RGB"))
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


if __name__ == "__main__":
    main()
