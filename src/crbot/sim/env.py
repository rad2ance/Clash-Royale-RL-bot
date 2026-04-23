from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SimConfig:
    max_steps: int = 900
    step_seconds: float = 0.2
    max_elixir: float = 10.0
    elixir_regen_per_step: float = 0.1
    king_hp: float = 4000.0
    princess_hp: float = 2500.0
    hand_size: int = 4
    n_cards: int = 12
    grid_w: int = 8
    grid_h: int = 14


class CrLikeSimEnv(gym.Env):
    """
    Abstract Clash Royale-like simulator.

    This is intentionally simplified. It gives us:
    - fixed API for RL/IL experiments
    - action decoding (no-op or card+placement)
    - sparse-ish reward from tower damage exchange
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SimConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.cfg = config or SimConfig()
        self.rng = np.random.default_rng(seed)

        self.noop_action = 0
        self.grid_size = self.cfg.grid_w * self.cfg.grid_h
        self.actions_per_card = self.grid_size
        self.n_actions = 1 + self.cfg.hand_size * self.actions_per_card

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
                "hand_ids": spaces.Box(
                    low=0.0, high=float(self.cfg.n_cards), shape=(self.cfg.hand_size,), dtype=np.float32
                ),
                "hand_costs": spaces.Box(low=0.0, high=10.0, shape=(self.cfg.hand_size,), dtype=np.float32),
            }
        )

        self.step_count = 0
        self.time_left = 180.0
        self.elixir = 5.0
        self.own_king_hp = self.cfg.king_hp
        self.enemy_king_hp = self.cfg.king_hp
        self.own_princess_hps = np.full(2, self.cfg.princess_hp, dtype=np.float32)
        self.enemy_princess_hps = np.full(2, self.cfg.princess_hp, dtype=np.float32)
        self.hand_ids = np.zeros(self.cfg.hand_size, dtype=np.int32)
        self.hand_costs = np.zeros(self.cfg.hand_size, dtype=np.float32)
        self._draw_initial_hand()

    def _draw_initial_hand(self) -> None:
        self.hand_ids = self.rng.integers(0, self.cfg.n_cards, size=self.cfg.hand_size, dtype=np.int32)
        self.hand_costs = self.rng.integers(1, 7, size=self.cfg.hand_size).astype(np.float32)

    def _draw_replacement_card(self, slot: int) -> None:
        self.hand_ids[slot] = int(self.rng.integers(0, self.cfg.n_cards))
        self.hand_costs[slot] = float(self.rng.integers(1, 7))

    def _get_obs(self) -> dict[str, np.ndarray]:
        global_vec = np.array(
            [
                self.time_left / 180.0,
                self.elixir / self.cfg.max_elixir,
                self.own_king_hp / self.cfg.king_hp,
                self.enemy_king_hp / self.cfg.king_hp,
                self.own_princess_hps.mean() / self.cfg.princess_hp,
                self.enemy_princess_hps.mean() / self.cfg.princess_hp,
                float(self.step_count) / float(self.cfg.max_steps),
                1.0 if self.time_left <= 60.0 else 0.0,  # double-elixir-ish phase
            ],
            dtype=np.float32,
        )
        return {
            "global": global_vec,
            "hand_ids": self.hand_ids.astype(np.float32),
            "hand_costs": self.hand_costs.astype(np.float32),
        }

    def _decode_action(self, action: int) -> tuple[int, int, int] | None:
        if action == self.noop_action:
            return None
        idx = action - 1
        slot = idx // self.actions_per_card
        rem = idx % self.actions_per_card
        x = rem % self.cfg.grid_w
        y = rem // self.cfg.grid_w
        return slot, x, y

    def _alive(self) -> bool:
        return (
            self.own_king_hp > 0.0
            and self.enemy_king_hp > 0.0
            and self.step_count < self.cfg.max_steps
            and self.time_left > 0.0
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.time_left = 180.0
        self.elixir = 5.0
        self.own_king_hp = self.cfg.king_hp
        self.enemy_king_hp = self.cfg.king_hp
        self.own_princess_hps[:] = self.cfg.princess_hp
        self.enemy_princess_hps[:] = self.cfg.princess_hp
        self._draw_initial_hand()
        return self._get_obs(), {}

    def step(self, action: int):
        decoded = self._decode_action(int(action))
        legal_action = True
        spent_elixir = 0.0
        damage_to_enemy = 0.0
        damage_to_self = 0.0

        if decoded is not None:
            slot, x, y = decoded
            if slot < 0 or slot >= self.cfg.hand_size:
                legal_action = False
            else:
                cost = float(self.hand_costs[slot])
                if self.elixir + 1e-6 < cost:
                    legal_action = False
                else:
                    spent_elixir = cost
                    self.elixir = max(0.0, self.elixir - cost)
                    board_factor = 1.0 - abs((x / max(1, self.cfg.grid_w - 1)) - 0.5)
                    range_factor = 0.7 + 0.6 * (y / max(1, self.cfg.grid_h - 1))

                    damage_to_enemy = float((8.0 + 5.0 * cost) * board_factor * range_factor)
                    enemy_chip = self.rng.uniform(0.8, 1.2)
                    damage_to_enemy *= enemy_chip
                    damage_to_self = float(self.rng.uniform(0.0, 3.5) * cost)

                    self.enemy_king_hp = max(0.0, self.enemy_king_hp - 0.55 * damage_to_enemy)
                    self.enemy_princess_hps -= 0.45 * damage_to_enemy
                    self.enemy_princess_hps = np.clip(self.enemy_princess_hps, 0.0, self.cfg.princess_hp)

                    self.own_king_hp = max(0.0, self.own_king_hp - 0.55 * damage_to_self)
                    self.own_princess_hps -= 0.45 * damage_to_self
                    self.own_princess_hps = np.clip(self.own_princess_hps, 0.0, self.cfg.princess_hp)

                    self._draw_replacement_card(slot)

        if not legal_action:
            damage_to_self += 2.0
            self.own_king_hp = max(0.0, self.own_king_hp - 1.0)

        self.elixir = min(self.cfg.max_elixir, self.elixir + self.cfg.elixir_regen_per_step)
        self.time_left = max(0.0, self.time_left - self.cfg.step_seconds)
        self.step_count += 1

        reward = (damage_to_enemy - 0.6 * damage_to_self) / 40.0
        if not legal_action:
            reward -= 0.05

        terminated = not self._alive()
        truncated = False
        if terminated:
            if self.enemy_king_hp <= 0.0 and self.own_king_hp > 0.0:
                reward += 2.0
            elif self.own_king_hp <= 0.0 and self.enemy_king_hp > 0.0:
                reward -= 2.0
            else:
                hp_delta = (self.enemy_king_hp + self.enemy_princess_hps.sum()) - (
                    self.own_king_hp + self.own_princess_hps.sum()
                )
                reward += float(np.clip(-hp_delta / 3000.0, -1.0, 1.0))

        info = {
            "legal_action": legal_action,
            "spent_elixir": spent_elixir,
            "damage_to_enemy": damage_to_enemy,
            "damage_to_self": damage_to_self,
            "decoded_action": decoded,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def flatten_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Convert dict observation into flat vector for simple MLP policies."""
    parts = [obs["global"].ravel(), obs["hand_ids"].ravel(), obs["hand_costs"].ravel()]
    return np.concatenate(parts, axis=0).astype(np.float32)

