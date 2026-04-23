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
    spell_card_count: int = 3
    building_card_count: int = 2
    grid_w: int = 8
    grid_h: int = 14
    # Actions can only deploy on the player's side of the arena.
    # With y=0 at top and y increasing downward, this defaults to bottom half.
    deploy_min_y: int | None = None
    # River rows are blocked for non-spell placements.
    river_top_y: int | None = None
    river_bottom_y: int | None = None
    # Building placements are constrained to lanes near bridges.
    bridge_xs: tuple[int, ...] = (2, 5)
    bridge_lane_half_width: int = 1


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
        self.deploy_min_y = self.cfg.deploy_min_y if self.cfg.deploy_min_y is not None else self.cfg.grid_h // 2
        default_river_top = max(0, self.deploy_min_y - 1)
        self.river_top_y = self.cfg.river_top_y if self.cfg.river_top_y is not None else default_river_top
        self.river_bottom_y = (
            self.cfg.river_bottom_y if self.cfg.river_bottom_y is not None else min(self.cfg.grid_h - 1, self.deploy_min_y)
        )

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

    def _is_spell_card(self, card_id: int) -> bool:
        return 0 <= int(card_id) < self.cfg.spell_card_count

    def _is_building_card(self, card_id: int) -> bool:
        start = self.cfg.spell_card_count
        end = self.cfg.spell_card_count + self.cfg.building_card_count
        return start <= int(card_id) < end

    def _is_river_row(self, y: int) -> bool:
        lo = min(self.river_top_y, self.river_bottom_y)
        hi = max(self.river_top_y, self.river_bottom_y)
        return lo <= y <= hi

    def _is_bridge_lane_x(self, x: int) -> bool:
        half = max(0, int(self.cfg.bridge_lane_half_width))
        for bx in self.cfg.bridge_xs:
            if abs(int(x) - int(bx)) <= half:
                return True
        return False

    def get_legal_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask over the discrete action space.

        Action 0 (noop) is always legal. Card-placement actions are legal only
        when their hand slot can be afforded with the current elixir.
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.noop_action] = True
        affordable_slots = self.hand_costs <= (self.elixir + 1e-6)
        own_side_mask = np.zeros(self.actions_per_card, dtype=bool)
        building_mask = np.zeros(self.actions_per_card, dtype=bool)
        for y in range(self.cfg.grid_h):
            if y < self.deploy_min_y:
                continue
            if self._is_river_row(y):
                continue
            row_start = y * self.cfg.grid_w
            row_end = row_start + self.cfg.grid_w
            own_side_mask[row_start:row_end] = True
            for x in range(self.cfg.grid_w):
                if self._is_bridge_lane_x(x):
                    building_mask[row_start + x] = True
        all_arena_mask = np.ones(self.actions_per_card, dtype=bool)

        for slot in range(self.cfg.hand_size):
            if not affordable_slots[slot]:
                continue
            start = 1 + slot * self.actions_per_card
            stop = start + self.actions_per_card
            card_id = int(self.hand_ids[slot])
            if self._is_spell_card(card_id):
                slot_mask = all_arena_mask
            elif self._is_building_card(card_id):
                slot_mask = building_mask
            else:
                slot_mask = own_side_mask
            mask[start:stop] = slot_mask
        return mask

    def action_masks(self) -> np.ndarray:
        """
        Compatibility alias used by sb3-contrib MaskablePPO wrappers.
        """
        return self.get_legal_action_mask()

    def is_action_legal(self, action: int) -> bool:
        if action < 0 or action >= self.n_actions:
            return False
        return bool(self.get_legal_action_mask()[action])

    def sample_legal_action(self) -> int:
        mask = self.get_legal_action_mask()
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return self.noop_action
        return int(self.rng.choice(legal))

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
        return self._get_obs(), {"legal_action_mask": self.get_legal_action_mask()}

    def step(self, action: int):
        action_i = int(action)
        decoded = self._decode_action(action_i)
        legal_action = self.is_action_legal(action_i)
        spent_elixir = 0.0
        damage_to_enemy = 0.0
        damage_to_self = 0.0

        if legal_action and decoded is not None:
            slot, x, y = decoded
            cost = float(self.hand_costs[slot])
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
            "illegal_reason": None if legal_action else "masked_or_unaffordable_action",
            "spent_elixir": spent_elixir,
            "damage_to_enemy": damage_to_enemy,
            "damage_to_self": damage_to_self,
            "decoded_action": decoded,
            "legal_action_mask": self.get_legal_action_mask(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def flatten_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Convert dict observation into flat vector for simple MLP policies."""
    parts = [obs["global"].ravel(), obs["hand_ids"].ravel(), obs["hand_costs"].ravel()]
    return np.concatenate(parts, axis=0).astype(np.float32)
