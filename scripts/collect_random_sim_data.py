from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crbot.data import EpisodeBatch, save_episode
from crbot.sim import CrLikeSimEnv, flatten_observation


def collect_episode(env: CrLikeSimEnv, max_steps: int) -> EpisodeBatch:
    obs, _ = env.reset()
    obs_buf: list[np.ndarray] = []
    act_buf: list[int] = []
    rew_buf: list[float] = []
    done_buf: list[bool] = []

    for _ in range(max_steps):
        flat = flatten_observation(obs)
        action = int(env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = env.step(action)

        obs_buf.append(flat)
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(bool(terminated or truncated))

        obs = next_obs
        if terminated or truncated:
            break

    return EpisodeBatch(
        observations=np.stack(obs_buf).astype(np.float32),
        actions=np.array(act_buf, dtype=np.int64),
        rewards=np.array(rew_buf, dtype=np.float32),
        dones=np.array(done_buf, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random trajectories from CR-like simulator.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--out", type=str, default="data/sim_random")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = CrLikeSimEnv()
    for i in range(args.episodes):
        ep = collect_episode(env, max_steps=args.max_steps)
        save_episode(out_dir / f"episode_{i:05d}.npz", ep)
        if (i + 1) % 20 == 0:
            print(f"[collect] saved {i + 1}/{args.episodes} episodes")

    print(f"[done] wrote trajectories to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

