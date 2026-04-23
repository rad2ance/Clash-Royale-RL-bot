from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation

from crbot.models import BcPolicy
from crbot.sim import CrLikeSimEnv, flatten_observation


@dataclass(frozen=True)
class EpisodeEval:
    episode: int
    total_reward: float
    steps: int
    illegal_actions: int
    win: bool
    loss: bool


def summarize(records: list[EpisodeEval]) -> dict[str, float]:
    if not records:
        raise ValueError("No episode records to summarize.")
    rewards = np.array([r.total_reward for r in records], dtype=np.float64)
    lengths = np.array([r.steps for r in records], dtype=np.float64)
    total_steps = max(1, int(lengths.sum()))
    total_illegal = int(sum(r.illegal_actions for r in records))
    wins = sum(1 for r in records if r.win)
    losses = sum(1 for r in records if r.loss)
    draws = len(records) - wins - losses
    return {
        "episodes": float(len(records)),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_episode_len": float(lengths.mean()),
        "illegal_action_rate": float(total_illegal / total_steps),
        "win_rate": float(wins / len(records)),
        "loss_rate": float(losses / len(records)),
        "draw_rate": float(draws / len(records)),
    }


def action_from_bc(model: BcPolicy, obs: dict[str, np.ndarray], device: str) -> int:
    x = torch.from_numpy(flatten_observation(obs)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        action = int(torch.argmax(logits, dim=1).item())
    return action


def action_from_bc_masked(model: BcPolicy, obs: dict[str, np.ndarray], mask: np.ndarray, device: str) -> int:
    x = torch.from_numpy(flatten_observation(obs)).unsqueeze(0).to(device)
    m = torch.from_numpy(mask.astype(bool)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        logits = logits.masked_fill(~m, -1e9)
        action = int(torch.argmax(logits, dim=1).item())
    return action


def eval_random(episodes: int, max_steps: int, seed: int) -> list[EpisodeEval]:
    env = CrLikeSimEnv(seed=seed)
    records: list[EpisodeEval] = []
    for ep in range(episodes):
        _, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        illegal = 0
        steps = 0
        while steps < max_steps:
            action = env.sample_legal_action()
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            illegal += 0 if bool(info.get("legal_action", True)) else 1
            steps += 1
            if terminated or truncated:
                break
        win = bool(env.enemy_king_hp <= 0.0 and env.own_king_hp > 0.0)
        loss = bool(env.own_king_hp <= 0.0 and env.enemy_king_hp > 0.0)
        records.append(EpisodeEval(ep, float(total_reward), steps, illegal, win, loss))
    return records


def eval_bc(checkpoint: str, episodes: int, max_steps: int, seed: int, masked: bool) -> list[EpisodeEval]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint, map_location=device)
    model = BcPolicy(
        obs_dim=int(payload["obs_dim"]),
        n_actions=int(payload["n_actions"]),
        hidden_dim=int(payload.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    env = CrLikeSimEnv(seed=seed)
    records: list[EpisodeEval] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        illegal = 0
        steps = 0
        while steps < max_steps:
            if masked:
                action = action_from_bc_masked(model, obs, env.get_legal_action_mask(), device)
            else:
                action = action_from_bc(model, obs, device)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            illegal += 0 if bool(info.get("legal_action", True)) else 1
            steps += 1
            if terminated or truncated:
                break
        win = bool(env.enemy_king_hp <= 0.0 and env.own_king_hp > 0.0)
        loss = bool(env.own_king_hp <= 0.0 and env.enemy_king_hp > 0.0)
        records.append(EpisodeEval(ep, float(total_reward), steps, illegal, win, loss))
    return records


def eval_ppo(checkpoint: str, episodes: int, max_steps: int, seed: int, masked: bool) -> list[EpisodeEval]:
    env = FlattenObservation(CrLikeSimEnv(seed=seed))

    if masked:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(checkpoint)
    else:
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint)

    records: list[EpisodeEval] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        illegal = 0
        steps = 0
        while steps < max_steps:
            if masked:
                action, _ = model.predict(obs, deterministic=True, action_masks=env.unwrapped.action_masks())
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            illegal += 0 if bool(info.get("legal_action", True)) else 1
            steps += 1
            if terminated or truncated:
                break
        base_env = env.unwrapped
        win = bool(base_env.enemy_king_hp <= 0.0 and base_env.own_king_hp > 0.0)
        loss = bool(base_env.own_king_hp <= 0.0 and base_env.enemy_king_hp > 0.0)
        records.append(EpisodeEval(ep, float(total_reward), steps, illegal, win, loss))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy checkpoints on CR-like simulator.")
    parser.add_argument("--policy", choices=["random", "bc", "bc-mask", "ppo", "ppo-mask"], default="random")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint for bc/ppo variants.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/eval/policy_eval.json")
    args = parser.parse_args()

    if args.policy != "random" and not args.checkpoint:
        raise ValueError("--checkpoint is required for bc/ppo/ppo-mask policies.")

    if args.policy == "random":
        records = eval_random(args.episodes, args.max_steps, args.seed)
    elif args.policy == "bc":
        records = eval_bc(args.checkpoint, args.episodes, args.max_steps, args.seed, masked=False)
    elif args.policy == "bc-mask":
        records = eval_bc(args.checkpoint, args.episodes, args.max_steps, args.seed, masked=True)
    elif args.policy == "ppo":
        records = eval_ppo(args.checkpoint, args.episodes, args.max_steps, args.seed, masked=False)
    else:
        records = eval_ppo(args.checkpoint, args.episodes, args.max_steps, args.seed, masked=True)

    summary = summarize(records)
    output: dict[str, Any] = {
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "summary": summary,
        "episodes": [asdict(r) for r in records],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[done] saved evaluation: {out_path.resolve()}")
    print(
        "[summary] "
        f"mean_reward={summary['mean_reward']:.3f} "
        f"win_rate={summary['win_rate']:.3f} "
        f"illegal_action_rate={summary['illegal_action_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
