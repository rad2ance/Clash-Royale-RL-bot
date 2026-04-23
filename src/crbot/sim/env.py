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
    enemy_spawn_chance: float = 0.08
    troop_lifetime_steps: int = 18
    building_lifetime_steps: int = 28
    troop_attack_range_cells: int = 2
    building_attack_range_cells: int = 4
    attack_cooldown_steps: int = 3


@dataclass(frozen=True)
class CardMeta:
    card_id: int
    name: str
    elixir_cost: int
    card_type: str  # "spell" | "building" | "troop"
    target_type: str  # "ground" | "air" | "any" | "area"
    can_hit_air: bool


@dataclass
class ActiveUnit:
    x: int
    y: int
    hp: float
    dps: float
    ttl: int
    card_type: str
    target_type: str
    can_hit_air: bool
    is_air: bool
    is_enemy: bool
    attack_range: int = 2
    attack_cooldown_steps: int = 3
    cooldown_remaining: int = 0
    hit_damage: float = 1.0


class CrLikeSimEnv(gym.Env):
    """
    Abstract Clash Royale-like simulator.

    This is intentionally simplified. It gives us:
    - fixed API for RL/IL experiments
    - action decoding (no-op or card+placement)
    - sparse-ish reward from tower damage exchange
    """

    metadata = {"render_modes": ["rgb_array"]}

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
        self.card_catalog = self._build_default_card_catalog()

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
        self.own_units: list[ActiveUnit] = []
        self.enemy_units: list[ActiveUnit] = []
        self._last_decoded_action: tuple[int, int, int] | None = None
        self._draw_initial_hand()

    def _build_default_card_catalog(self) -> dict[int, CardMeta]:
        """
        Construct deterministic metadata for each card id.

        This keeps the simulator lightweight while ensuring card ids map to
        stable costs and semantics across resets and episodes.
        """
        catalog: dict[int, CardMeta] = {}
        for cid in range(self.cfg.n_cards):
            if cid < self.cfg.spell_card_count:
                cost = 2 + (cid % 3)
                card_type = "spell"
                target_type = "area"
                can_hit_air = True
            elif cid < self.cfg.spell_card_count + self.cfg.building_card_count:
                rel = cid - self.cfg.spell_card_count
                cost = 4 + (rel % 2)
                card_type = "building"
                target_type = "ground"
                can_hit_air = False
            else:
                rel = cid - (self.cfg.spell_card_count + self.cfg.building_card_count)
                cost = 2 + (rel % 5)
                card_type = "troop"
                can_hit_air = bool(rel % 2)
                target_type = "any" if can_hit_air else "ground"

            catalog[cid] = CardMeta(
                card_id=cid,
                name=f"card_{cid:02d}",
                elixir_cost=int(cost),
                card_type=card_type,
                target_type=target_type,
                can_hit_air=can_hit_air,
            )
        return catalog

    def get_card_meta(self, card_id: int) -> CardMeta:
        cid = int(card_id)
        if cid not in self.card_catalog:
            raise KeyError(f"Unknown card id: {cid}")
        return self.card_catalog[cid]

    def _sync_hand_costs_from_ids(self) -> None:
        for slot in range(self.cfg.hand_size):
            cid = int(self.hand_ids[slot])
            self.hand_costs[slot] = float(self.get_card_meta(cid).elixir_cost)

    def _draw_initial_hand(self) -> None:
        self.hand_ids = self.rng.integers(0, self.cfg.n_cards, size=self.cfg.hand_size, dtype=np.int32)
        self._sync_hand_costs_from_ids()

    def _draw_replacement_card(self, slot: int) -> None:
        self.hand_ids[slot] = int(self.rng.integers(0, self.cfg.n_cards))
        self.hand_costs[slot] = float(self.get_card_meta(int(self.hand_ids[slot])).elixir_cost)

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
        return self.get_card_meta(card_id).card_type == "spell"

    def _is_building_card(self, card_id: int) -> bool:
        return self.get_card_meta(card_id).card_type == "building"

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

    def _nearest_bridge_x(self, x: int) -> int:
        if not self.cfg.bridge_xs:
            return int(np.clip(x, 0, self.cfg.grid_w - 1))
        target = min(self.cfg.bridge_xs, key=lambda bx: abs(int(x) - int(bx)))
        return int(np.clip(target, 0, self.cfg.grid_w - 1))

    def _combat_profile(self, card: CardMeta) -> tuple[float, float, float]:
        """
        Return (offense_multiplier, self_damage_multiplier, king_damage_share).
        """
        offense_mult = 1.0
        self_damage_mult = 1.0
        king_share = 0.55

        if card.card_type == "spell":
            offense_mult *= 1.15
            self_damage_mult *= 0.25
            king_share = 0.45
        elif card.card_type == "building":
            offense_mult *= 0.85
            self_damage_mult *= 0.55
            king_share = 0.35
        else:
            king_share = 0.55

        if card.target_type == "ground":
            offense_mult *= 0.95
            king_share = min(king_share, 0.4)
        elif card.target_type == "any":
            offense_mult *= 1.05
            king_share = max(king_share, 0.6)
        elif card.target_type == "area":
            offense_mult *= 1.1

        if card.can_hit_air:
            self_damage_mult *= 0.9

        return offense_mult, self_damage_mult, king_share

    def _unit_profile(self, card: CardMeta, cost: float) -> tuple[float, float, int, int, int, float]:
        cooldown = max(1, int(self.cfg.attack_cooldown_steps))
        if card.card_type == "building":
            dps = 3.2 * cost
            hp = 32.0 * cost
            ttl = self.cfg.building_lifetime_steps
            attack_range = self.cfg.building_attack_range_cells
        else:
            dps = 4.0 * cost
            hp = 18.0 * cost
            ttl = self.cfg.troop_lifetime_steps
            attack_range = self.cfg.troop_attack_range_cells
        if card.can_hit_air:
            dps *= 1.08
        if card.target_type == "ground":
            hp *= 1.12
        hit_damage = dps * self.cfg.step_seconds * float(cooldown)
        return dps, hp, ttl, int(attack_range), cooldown, float(hit_damage)

    def _spawn_friendly_unit(self, card: CardMeta, x: int, y: int, cost: float) -> None:
        if card.card_type == "spell":
            return
        dps, hp, ttl, attack_range, cooldown, hit_damage = self._unit_profile(card, cost)
        self.own_units.append(
            ActiveUnit(
                x=int(np.clip(x, 0, self.cfg.grid_w - 1)),
                y=int(np.clip(y, 0, self.cfg.grid_h - 1)),
                hp=float(hp),
                dps=float(dps),
                ttl=int(ttl),
                card_type=card.card_type,
                target_type=card.target_type,
                can_hit_air=card.can_hit_air,
                is_air=False,
                is_enemy=False,
                attack_range=attack_range,
                attack_cooldown_steps=cooldown,
                cooldown_remaining=0,
                hit_damage=hit_damage,
            )
        )

    def _spawn_enemy_unit(self) -> None:
        if self.rng.uniform() >= self.cfg.enemy_spawn_chance:
            return
        x = int(self.rng.integers(0, self.cfg.grid_w))
        y = int(self.rng.integers(0, max(1, self.deploy_min_y - 1)))
        cost = float(self.rng.integers(2, 6))
        self.enemy_units.append(
            ActiveUnit(
                x=x,
                y=y,
                hp=16.0 * cost,
                dps=3.8 * cost,
                ttl=max(6, int(self.cfg.troop_lifetime_steps * 0.7)),
                card_type="troop",
                target_type="ground",
                can_hit_air=False,
                is_air=False,
                is_enemy=True,
                attack_range=self.cfg.troop_attack_range_cells,
                attack_cooldown_steps=max(1, int(self.cfg.attack_cooldown_steps)),
                cooldown_remaining=0,
                hit_damage=float(3.8 * cost * self.cfg.step_seconds * max(1, int(self.cfg.attack_cooldown_steps))),
            )
        )

    def _can_target(self, attacker: ActiveUnit, target: ActiveUnit) -> bool:
        if target.is_air and not attacker.can_hit_air:
            return False
        if attacker.target_type == "ground" and target.is_air:
            return False
        return True

    def _closest_target(self, attacker: ActiveUnit, candidates: list[ActiveUnit]) -> ActiveUnit | None:
        in_range: list[ActiveUnit] = []
        for target in candidates:
            if not self._can_target(attacker, target):
                continue
            if abs(attacker.x - target.x) + abs(attacker.y - target.y) <= max(1, int(attacker.attack_range)):
                in_range.append(target)
        if not in_range:
            return None
        return min(in_range, key=lambda u: (abs(attacker.x - u.x) + abs(attacker.y - u.y), u.hp))

    def _process_unit_duels(self) -> tuple[set[int], set[int]]:
        attacked_own: set[int] = set()
        attacked_enemy: set[int] = set()
        if not self.own_units or not self.enemy_units:
            return attacked_own, attacked_enemy
        for own in self.own_units:
            if own.cooldown_remaining > 0:
                continue
            target = self._closest_target(own, self.enemy_units)
            if target is not None:
                target.hp -= own.hit_damage
                own.cooldown_remaining = own.attack_cooldown_steps
                attacked_own.add(id(own))
        for enemy in self.enemy_units:
            if enemy.cooldown_remaining > 0:
                continue
            target = self._closest_target(enemy, self.own_units)
            if target is not None:
                target.hp -= enemy.hit_damage
                enemy.cooldown_remaining = enemy.attack_cooldown_steps
                attacked_enemy.add(id(enemy))
        return attacked_own, attacked_enemy

    def _in_enemy_tower_zone(self, unit: ActiveUnit) -> bool:
        return unit.y <= self.river_top_y

    def _in_own_tower_zone(self, unit: ActiveUnit) -> bool:
        return unit.y >= self.river_bottom_y

    def _process_tower_attacks(self, attacked_own: set[int], attacked_enemy: set[int]) -> tuple[float, float]:
        damage_to_enemy = 0.0
        damage_to_self = 0.0
        own_king_share = 0.55
        enemy_king_share = 0.55

        for unit in self.own_units:
            if id(unit) in attacked_own or unit.cooldown_remaining > 0 or not self._in_enemy_tower_zone(unit):
                continue
            pressure = unit.hit_damage
            if unit.card_type == "building":
                pressure *= 0.75
            damage_to_enemy += pressure
            self.enemy_king_hp = max(0.0, self.enemy_king_hp - own_king_share * pressure)
            self.enemy_princess_hps -= (1.0 - own_king_share) * pressure
            unit.cooldown_remaining = unit.attack_cooldown_steps

        for unit in self.enemy_units:
            if id(unit) in attacked_enemy or unit.cooldown_remaining > 0 or not self._in_own_tower_zone(unit):
                continue
            pressure = unit.hit_damage
            damage_to_self += pressure
            self.own_king_hp = max(0.0, self.own_king_hp - enemy_king_share * pressure)
            self.own_princess_hps -= (1.0 - enemy_king_share) * pressure
            unit.cooldown_remaining = unit.attack_cooldown_steps

        self.enemy_princess_hps = np.clip(self.enemy_princess_hps, 0.0, self.cfg.princess_hp)
        self.own_princess_hps = np.clip(self.own_princess_hps, 0.0, self.cfg.princess_hp)
        return float(damage_to_enemy), float(damage_to_self)

    def _advance_units(self) -> tuple[float, float]:
        self._spawn_enemy_unit()
        for unit in self.own_units:
            unit.cooldown_remaining = max(0, unit.cooldown_remaining - 1)
        for unit in self.enemy_units:
            unit.cooldown_remaining = max(0, unit.cooldown_remaining - 1)

        attacked_own, attacked_enemy = self._process_unit_duels()
        dmg_enemy, dmg_self = self._process_tower_attacks(attacked_own, attacked_enemy)

        for unit in self.own_units:
            unit.ttl -= 1
            unit.hp -= 0.2
            if unit.card_type != "building":
                next_y = max(0, unit.y - 1)
                if self._is_river_row(next_y) and not self._is_bridge_lane_x(unit.x):
                    target_x = self._nearest_bridge_x(unit.x)
                    if unit.x < target_x:
                        unit.x += 1
                    elif unit.x > target_x:
                        unit.x -= 1
                else:
                    unit.y = next_y
        for unit in self.enemy_units:
            unit.ttl -= 1
            unit.hp -= 0.2
            if unit.card_type != "building":
                next_y = min(self.cfg.grid_h - 1, unit.y + 1)
                if self._is_river_row(next_y) and not self._is_bridge_lane_x(unit.x):
                    target_x = self._nearest_bridge_x(unit.x)
                    if unit.x < target_x:
                        unit.x += 1
                    elif unit.x > target_x:
                        unit.x -= 1
                else:
                    unit.y = next_y

        self.own_units = [u for u in self.own_units if u.ttl > 0 and u.hp > 0.0]
        self.enemy_units = [u for u in self.enemy_units if u.ttl > 0 and u.hp > 0.0]
        return dmg_enemy, dmg_self

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
        self.own_units = []
        self.enemy_units = []
        self._last_decoded_action = None
        return self._get_obs(), {"legal_action_mask": self.get_legal_action_mask()}

    def step(self, action: int):
        action_i = int(action)
        decoded = self._decode_action(action_i)
        self._last_decoded_action = decoded
        legal_action = self.is_action_legal(action_i)
        spent_elixir = 0.0
        damage_to_enemy = 0.0
        damage_to_self = 0.0
        played_card_type: str | None = None

        if legal_action and decoded is not None:
            slot, x, y = decoded
            card = self.get_card_meta(int(self.hand_ids[slot]))
            played_card_type = card.card_type
            cost = float(self.hand_costs[slot])
            spent_elixir = cost
            self.elixir = max(0.0, self.elixir - cost)
            board_factor = 1.0 - abs((x / max(1, self.cfg.grid_w - 1)) - 0.5)
            range_factor = 0.7 + 0.6 * (y / max(1, self.cfg.grid_h - 1))
            offense_mult, self_damage_mult, king_share = self._combat_profile(card)

            damage_to_enemy = float((8.0 + 5.0 * cost) * board_factor * range_factor)
            damage_to_enemy *= offense_mult
            enemy_chip = self.rng.uniform(0.8, 1.2)
            damage_to_enemy *= enemy_chip
            damage_to_self = float(self.rng.uniform(0.0, 3.5) * cost * self_damage_mult)

            princess_share = 1.0 - king_share
            self.enemy_king_hp = max(0.0, self.enemy_king_hp - king_share * damage_to_enemy)
            self.enemy_princess_hps -= princess_share * damage_to_enemy
            self.enemy_princess_hps = np.clip(self.enemy_princess_hps, 0.0, self.cfg.princess_hp)

            self.own_king_hp = max(0.0, self.own_king_hp - 0.55 * damage_to_self)
            self.own_princess_hps -= 0.45 * damage_to_self
            self.own_princess_hps = np.clip(self.own_princess_hps, 0.0, self.cfg.princess_hp)

            self._spawn_friendly_unit(card=card, x=x, y=y, cost=cost)
            self._draw_replacement_card(slot)

        if not legal_action:
            damage_to_self += 2.0
            self.own_king_hp = max(0.0, self.own_king_hp - 1.0)

        ongoing_enemy, ongoing_self = self._advance_units()
        damage_to_enemy += ongoing_enemy
        damage_to_self += ongoing_self

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
            "card_type": played_card_type,
            "ongoing_damage_to_enemy": ongoing_enemy,
            "ongoing_damage_to_self": ongoing_self,
            "own_active_units": len(self.own_units),
            "enemy_active_units": len(self.enemy_units),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Render a lightweight top-down frame of the simulator state.
        Returns an RGB uint8 image with shape [H, W, 3].
        """
        cell = 18
        arena_w = self.cfg.grid_w * cell
        arena_h = self.cfg.grid_h * cell
        hand_h = 72
        status_h = 36
        width = arena_w
        height = status_h + arena_h + hand_h
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Status strip.
        img[:status_h, :, :] = np.array([24, 24, 28], dtype=np.uint8)
        time_ratio = np.clip(self.time_left / 180.0, 0.0, 1.0)
        elixir_ratio = np.clip(self.elixir / max(self.cfg.max_elixir, 1e-6), 0.0, 1.0)
        t_w = int((width - 20) * time_ratio)
        e_w = int((width - 20) * elixir_ratio)
        img[8:14, 10 : 10 + t_w, :] = np.array([84, 160, 255], dtype=np.uint8)
        img[20:28, 10 : 10 + e_w, :] = np.array([80, 220, 140], dtype=np.uint8)

        # Arena background.
        y0 = status_h
        y1 = status_h + arena_h
        img[y0:y1, :, :] = np.array([34, 128, 84], dtype=np.uint8)

        # Grid lines.
        grid_line = np.array([44, 146, 98], dtype=np.uint8)
        for gx in range(self.cfg.grid_w + 1):
            x = gx * cell
            img[y0:y1, max(0, x - 1) : min(width, x + 1), :] = grid_line
        for gy in range(self.cfg.grid_h + 1):
            y = y0 + gy * cell
            img[max(y0, y - 1) : min(y1, y + 1), :, :] = grid_line

        # River rows.
        river_lo = min(self.river_top_y, self.river_bottom_y)
        river_hi = max(self.river_top_y, self.river_bottom_y)
        for gy in range(river_lo, river_hi + 1):
            ry0 = y0 + gy * cell
            ry1 = min(y1, ry0 + cell)
            img[ry0:ry1, :, :] = np.array([36, 108, 188], dtype=np.uint8)

        # Bridge lanes.
        for bx in self.cfg.bridge_xs:
            bx0 = max(0, int((bx - self.cfg.bridge_lane_half_width) * cell))
            bx1 = min(width, int((bx + self.cfg.bridge_lane_half_width + 1) * cell))
            for gy in range(river_lo, river_hi + 1):
                ry0 = y0 + gy * cell
                ry1 = min(y1, ry0 + cell)
                img[ry0:ry1, bx0:bx1, :] = np.array([194, 162, 108], dtype=np.uint8)

        # Last action marker.
        if self._last_decoded_action is not None:
            _, ax, ay = self._last_decoded_action
            cx0 = ax * cell + 3
            cx1 = min(width, (ax + 1) * cell - 3)
            cy0 = y0 + ay * cell + 3
            cy1 = min(y1, y0 + (ay + 1) * cell - 3)
            img[cy0:cy1, cx0:cx1, :] = np.array([255, 82, 82], dtype=np.uint8)

        # Active unit markers.
        for unit in self.own_units:
            ux0 = unit.x * cell + 5
            ux1 = min(width, (unit.x + 1) * cell - 5)
            uy0 = y0 + unit.y * cell + 5
            uy1 = min(y1, y0 + (unit.y + 1) * cell - 5)
            img[uy0:uy1, ux0:ux1, :] = np.array([88, 232, 124], dtype=np.uint8)
        for unit in self.enemy_units:
            ux0 = unit.x * cell + 5
            ux1 = min(width, (unit.x + 1) * cell - 5)
            uy0 = y0 + unit.y * cell + 5
            uy1 = min(y1, y0 + (unit.y + 1) * cell - 5)
            img[uy0:uy1, ux0:ux1, :] = np.array([255, 124, 104], dtype=np.uint8)

        # Simple tower HP bars.
        own_hp = np.clip(self.own_king_hp / max(self.cfg.king_hp, 1e-6), 0.0, 1.0)
        enemy_hp = np.clip(self.enemy_king_hp / max(self.cfg.king_hp, 1e-6), 0.0, 1.0)
        own_w = int((width // 2 - 20) * own_hp)
        enemy_w = int((width // 2 - 20) * enemy_hp)
        img[y0 + arena_h - 14 : y0 + arena_h - 8, 10 : 10 + own_w, :] = np.array([80, 220, 140], dtype=np.uint8)
        ex0 = width // 2 + 10
        img[y0 + 8 : y0 + 14, ex0 : ex0 + enemy_w, :] = np.array([255, 110, 96], dtype=np.uint8)

        # Hand strip.
        hy0 = y1
        img[hy0:height, :, :] = np.array([40, 40, 46], dtype=np.uint8)
        slot_w = width // self.cfg.hand_size
        for s in range(self.cfg.hand_size):
            sx0 = s * slot_w
            sx1 = min(width, (s + 1) * slot_w)
            card = self.get_card_meta(int(self.hand_ids[s]))
            if card.card_type == "spell":
                base = np.array([76, 132, 255], dtype=np.uint8)
            elif card.card_type == "building":
                base = np.array([194, 162, 108], dtype=np.uint8)
            else:
                base = np.array([108, 206, 124], dtype=np.uint8)
            pad = 6
            img[hy0 + 8 : height - 24, sx0 + pad : sx1 - pad, :] = base
            mana = int(np.clip(self.hand_costs[s], 0, 10))
            mana_w = int((sx1 - sx0 - 2 * pad) * (mana / 10.0))
            img[height - 18 : height - 12, sx0 + pad : sx0 + pad + mana_w, :] = np.array([230, 230, 236], dtype=np.uint8)

        return img


def flatten_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Convert dict observation into flat vector for simple MLP policies."""
    parts = [obs["global"].ravel(), obs["hand_ids"].ravel(), obs["hand_costs"].ravel()]
    return np.concatenate(parts, axis=0).astype(np.float32)
