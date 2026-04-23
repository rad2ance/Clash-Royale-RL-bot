from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from crbot.sim import CrLikeSimEnv, SimConfig, flatten_observation


@dataclass(frozen=True)
class BenchCase:
    name: str
    grid_w: int
    grid_h: int
    observe_unit_density: bool
    obs_grid_w: int
    obs_grid_h: int


@dataclass(frozen=True)
class BenchResult:
    name: str
    total_steps: int
    elapsed_s: float
    steps_per_s: float


def run_case(case: BenchCase, episodes: int, max_steps: int, seed: int, flatten_obs: bool) -> BenchResult:
    cfg = SimConfig(
        grid_w=case.grid_w,
        grid_h=case.grid_h,
        observe_unit_density=case.observe_unit_density,
        obs_grid_w=case.obs_grid_w,
        obs_grid_h=case.obs_grid_h,
    )
    env = CrLikeSimEnv(config=cfg, seed=seed)
    total_steps = 0
    start = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            if flatten_obs:
                _ = flatten_observation(obs)
            action = env.sample_legal_action()
            obs, _, terminated, truncated, _ = env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
    elapsed = max(1e-9, time.perf_counter() - start)
    return BenchResult(
        name=case.name,
        total_steps=total_steps,
        elapsed_s=elapsed,
        steps_per_s=float(total_steps / elapsed),
    )


def default_cases() -> list[BenchCase]:
    return [
        BenchCase("base_8x14", grid_w=8, grid_h=14, observe_unit_density=False, obs_grid_w=4, obs_grid_h=7),
        BenchCase("high_16x28", grid_w=16, grid_h=28, observe_unit_density=False, obs_grid_w=4, obs_grid_h=7),
        BenchCase("xhigh_24x42", grid_w=24, grid_h=42, observe_unit_density=False, obs_grid_w=4, obs_grid_h=7),
        BenchCase("base_8x14+density", grid_w=8, grid_h=14, observe_unit_density=True, obs_grid_w=4, obs_grid_h=7),
        BenchCase("high_16x28+density", grid_w=16, grid_h=28, observe_unit_density=True, obs_grid_w=4, obs_grid_h=7),
    ]


def print_table(results: list[BenchResult]) -> None:
    if not results:
        print("[done] no benchmark results.")
        return
    header = f"{'case':<22} {'steps':>8} {'elapsed_s':>10} {'steps/s':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<22} {r.total_steps:>8} {r.elapsed_s:>10.3f} {r.steps_per_s:>12.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simulator throughput across resolution presets.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flatten-obs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include observation flattening cost to approximate training input path.",
    )
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    cases = default_cases()
    results = [run_case(c, args.episodes, args.max_steps, args.seed, args.flatten_obs) for c in cases]
    print_table(results)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "flatten_obs": bool(args.flatten_obs),
            "results": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[done] wrote benchmark json: {out.resolve()}")


if __name__ == "__main__":
    main()
