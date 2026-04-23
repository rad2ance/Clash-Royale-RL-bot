from __future__ import annotations

import argparse
from pathlib import Path

from gymnasium.wrappers import FlattenObservation

from crbot.sim import CrLikeSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on CR-like simulator.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out", type=str, default="checkpoints/ppo_sim")
    parser.add_argument(
        "--mask-actions",
        action="store_true",
        help="Use action masking via sb3-contrib MaskablePPO.",
    )
    args = parser.parse_args()

    if args.mask_actions:
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Maskable PPO requires sb3-contrib. Run: pip install -e '.[rl]'"
            ) from exc
    else:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("stable-baselines3 is not installed. Run: pip install -e '.[rl]'") from exc

    env = FlattenObservation(CrLikeSimEnv())
    if args.mask_actions:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker

        env = ActionMasker(env, action_mask_fn=lambda e: e.unwrapped.action_masks())
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    model.learn(total_timesteps=args.timesteps)

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"[done] saved PPO model: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
