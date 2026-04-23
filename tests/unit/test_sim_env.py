import numpy as np

from crbot.sim import CrLikeSimEnv, flatten_observation


def test_env_reset_and_step_shapes() -> None:
    env = CrLikeSimEnv()
    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
    assert "legal_action_mask" in info
    assert info["legal_action_mask"].shape == (env.n_actions,)
    assert set(obs.keys()) == {"global", "hand_ids", "hand_costs"}

    flat = flatten_observation(obs)
    assert flat.ndim == 1
    assert flat.shape[0] == 16

    next_obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
    assert set(next_obs.keys()) == {"global", "hand_ids", "hand_costs"}
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "legal_action" in step_info
    assert "illegal_reason" in step_info
    assert "card_type" in step_info
    assert "legal_action_mask" in step_info
    assert step_info["legal_action_mask"].shape == (env.n_actions,)

    next_flat = flatten_observation(next_obs)
    assert next_flat.shape == flat.shape
    assert np.isfinite(next_flat).all()


def test_legal_action_mask_respects_elixir() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)

    env.elixir = 0.0
    env.hand_costs[:] = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mask = env.get_legal_action_mask()
    assert bool(mask[env.noop_action]) is True
    assert bool(mask[1:].any()) is False

    env.elixir = env.cfg.max_elixir
    mask = env.get_legal_action_mask()
    assert bool(mask[env.noop_action]) is True
    assert bool(mask[1:].any()) is True


def test_sample_legal_action_returns_only_legal_actions() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)

    env.elixir = 1.0
    env.hand_costs[:] = np.array([6.0, 6.0, 6.0, 6.0], dtype=np.float32)
    samples = [env.sample_legal_action() for _ in range(25)]
    assert all(a == env.noop_action for a in samples)

    env.elixir = env.cfg.max_elixir
    env.hand_costs[:] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    samples = [env.sample_legal_action() for _ in range(100)]
    assert any(a != env.noop_action for a in samples)
    assert all(env.is_action_legal(a) for a in samples)


def test_legal_action_mask_enforces_deploy_side() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    env.hand_ids[:] = np.array([env.cfg.spell_card_count + 1] * env.cfg.hand_size, dtype=np.int32)

    slot = 0
    top_y = max(0, env.deploy_min_y - 1)
    bottom_y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    x = env.cfg.grid_w // 2

    top_action = 1 + slot * env.actions_per_card + top_y * env.cfg.grid_w + x
    bottom_action = 1 + slot * env.actions_per_card + bottom_y * env.cfg.grid_w + x

    assert env.is_action_legal(bottom_action) is True
    assert env.is_action_legal(top_action) is False


def test_spell_cards_can_target_full_arena() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    env.hand_ids[:] = np.array([0, env.cfg.spell_card_count + 1, env.cfg.spell_card_count + 1, env.cfg.spell_card_count + 1], dtype=np.int32)

    slot = 0  # spell card slot
    top_y = max(0, env.deploy_min_y - 1)
    x = env.cfg.grid_w // 2
    top_action = 1 + slot * env.actions_per_card + top_y * env.cfg.grid_w + x
    assert env.is_action_legal(top_action) is True


def test_action_masks_alias_matches_legal_mask() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    assert np.array_equal(env.action_masks(), env.get_legal_action_mask())


def test_building_cards_are_center_lane_constrained() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    building_id = env.cfg.spell_card_count
    env.hand_ids[:] = np.array([building_id, building_id, building_id, building_id], dtype=np.int32)
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    slot = 0
    y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    edge_x = 0
    center_x = env.cfg.grid_w // 2
    edge_action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + edge_x
    center_action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + center_x

    assert env.is_action_legal(edge_action) is False
    assert env.is_action_legal(center_action) is True


def test_river_rows_block_non_spell_placements() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    troop_id = env.cfg.spell_card_count + env.cfg.building_card_count + 1
    env.hand_ids[:] = np.array([troop_id, troop_id, troop_id, troop_id], dtype=np.int32)
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    slot = 0
    x = env.cfg.grid_w // 2
    river_y = env.river_bottom_y
    action = 1 + slot * env.actions_per_card + river_y * env.cfg.grid_w + x
    assert env.is_action_legal(action) is False


def test_spell_can_target_river_rows() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    env.hand_ids[:] = np.array([0, 0, 0, 0], dtype=np.int32)
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    slot = 0
    x = env.cfg.grid_w // 2
    river_y = env.river_top_y
    action = 1 + slot * env.actions_per_card + river_y * env.cfg.grid_w + x
    assert env.is_action_legal(action) is True


def test_hand_costs_match_card_metadata_after_reset() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    for slot in range(env.cfg.hand_size):
        cid = int(env.hand_ids[slot])
        expected = float(env.get_card_meta(cid).elixir_cost)
        assert float(env.hand_costs[slot]) == expected


def test_replacement_card_cost_matches_metadata() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    env.hand_ids[:] = np.array([0, 1, 2, 3], dtype=np.int32)
    env.hand_costs[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Play slot 0 legally to force replacement draw for that slot.
    slot = 0
    y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    x = env.cfg.grid_w // 2
    action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + x
    _, _, _, _, info = env.step(action)
    assert info["legal_action"] is True

    new_cid = int(env.hand_ids[slot])
    expected = float(env.get_card_meta(new_cid).elixir_cost)
    assert float(env.hand_costs[slot]) == expected


def _run_single_play(card_id: int, x: int, y: int):
    env = CrLikeSimEnv()
    env.reset(seed=123)
    env.elixir = env.cfg.max_elixir
    env.hand_ids[:] = np.array([card_id, card_id, card_id, card_id], dtype=np.int32)
    expected_cost = float(env.get_card_meta(card_id).elixir_cost)
    env.hand_costs[:] = np.array([expected_cost] * env.cfg.hand_size, dtype=np.float32)

    slot = 0
    action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + x
    _, _, _, _, info = env.step(action)
    return info


def test_spell_profile_has_more_enemy_damage_and_less_self_damage_than_troop() -> None:
    env = CrLikeSimEnv()
    y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    x = env.cfg.grid_w // 2
    spell_id = 2
    troop_id = 7
    assert env.get_card_meta(spell_id).elixir_cost == env.get_card_meta(troop_id).elixir_cost

    spell_info = _run_single_play(card_id=spell_id, x=x, y=y)
    troop_info = _run_single_play(card_id=troop_id, x=x, y=y)

    assert spell_info["card_type"] == "spell"
    assert troop_info["card_type"] == "troop"
    assert float(spell_info["damage_to_enemy"]) > float(troop_info["damage_to_enemy"])
    assert float(spell_info["damage_to_self"]) < float(troop_info["damage_to_self"])


def test_building_profile_has_lower_enemy_damage_than_troop_same_cost() -> None:
    env = CrLikeSimEnv()
    y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    x = env.cfg.grid_w // 2
    building_id = env.cfg.spell_card_count
    troop_id = 7
    assert env.get_card_meta(building_id).elixir_cost == env.get_card_meta(troop_id).elixir_cost

    building_info = _run_single_play(card_id=building_id, x=x, y=y)
    troop_info = _run_single_play(card_id=troop_id, x=x, y=y)

    assert building_info["card_type"] == "building"
    assert troop_info["card_type"] == "troop"
    assert float(building_info["damage_to_enemy"]) < float(troop_info["damage_to_enemy"])
