from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crbot.data import EpisodeBatch, save_episode
from crbot.sim import CrLikeSimEnv, flatten_observation


def collect_episode(env: CrLikeSimEnv, max_steps: int, allow_illegal_actions: bool = False) -> EpisodeBatch:
    obs, _ = env.reset()
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    rew_buf: list[float] = []
    done_buf: list[bool] = []
    mask_buf: list[np.ndarray] = []

    for _ in range(max_steps):
        flat = flatten_observation(obs)
        legal_mask = env.get_legal_action_mask()
        if allow_illegal_actions:
            action = int(env.action_space.sample())
        else:
            action = env.sample_legal_action()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        obs_buf.append(flat)
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(bool(terminated or truncated))
        mask_buf.append(legal_mask.astype(bool))

        obs = next_obs
        if terminated or truncated:
            break

    return EpisodeBatch(
        observations=np.stack(obs_buf).astype(np.float32),
        actions=np.array(act_buf, dtype=np.int64),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=bool),
        action_masks=np.stack(mask_buf).astype(bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random trajectories from CR-like simulator.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--out", type=str, default="data/sim_random")
    parser.add_argument(
        "--allow-illegal-actions",
        action="store_true",
        help="Sample uniformly from all actions instead of only currently legal actions.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = CrLikeSimEnv()
    for i in range(args.episodes):
        ep = collect_episode(env, max_steps=args.max_steps, allow_illegal_actions=args.allow_illegal_actions)
        save_episode(out_dir / f"episode_{i:05d}.npz", ep)
        if (i + 1) % 20 == 0:
            print(f"[collect] saved {i + 1}/{args.episodes} episodes")

    print(f"[done] wrote trajectories to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
