import numpy as np

from crbot.sim import ActiveUnit, CrLikeSimEnv, SimConfig, flatten_observation


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
    assert "ongoing_damage_to_enemy" in step_info
    assert "ongoing_damage_to_self" in step_info
    assert "own_active_units" in step_info
    assert "enemy_active_units" in step_info
    assert "pending_projectiles" in step_info
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
    troop_id = env.cfg.spell_card_count + env.cfg.building_card_count + 1
    env.hand_ids[:] = np.array([troop_id] * env.cfg.hand_size, dtype=np.int32)

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
    bridge_x = env._nearest_bridge_x(env.cfg.grid_w // 2)
    edge_action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + edge_x
    bridge_action = 1 + slot * env.actions_per_card + y * env.cfg.grid_w + bridge_x

    assert env.is_action_legal(edge_action) is False
    assert env.is_action_legal(bridge_action) is True


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
    x = env._nearest_bridge_x(env.cfg.grid_w // 2)
    building_id = env.cfg.spell_card_count
    troop_id = 7
    assert env.get_card_meta(building_id).elixir_cost == env.get_card_meta(troop_id).elixir_cost

    building_info = _run_single_play(card_id=building_id, x=x, y=y)
    troop_info = _run_single_play(card_id=troop_id, x=x, y=y)

    assert building_info["card_type"] == "building"
    assert troop_info["card_type"] == "troop"
    assert float(building_info["damage_to_enemy"]) < float(troop_info["damage_to_enemy"])


def test_render_returns_rgb_array() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_troop_play_spawns_friendly_active_unit() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    troop_id = env.cfg.spell_card_count + env.cfg.building_card_count + 1
    env.hand_ids[:] = np.array([troop_id, troop_id, troop_id, troop_id], dtype=np.int32)
    env._sync_hand_costs_from_ids()

    y = min(env.cfg.grid_h - 1, env.deploy_min_y + 1)
    x = env.cfg.grid_w // 2
    action = 1 + y * env.cfg.grid_w + x
    _, _, _, _, info = env.step(action)

    assert info["legal_action"] is True
    assert int(info["own_active_units"]) >= 1


def test_spell_play_does_not_spawn_friendly_unit() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    env.elixir = env.cfg.max_elixir
    spell_id = 0
    env.hand_ids[:] = np.array([spell_id, spell_id, spell_id, spell_id], dtype=np.int32)
    env._sync_hand_costs_from_ids()

    y = max(0, env.deploy_min_y - 1)
    x = env.cfg.grid_w // 2
    action = 1 + y * env.cfg.grid_w + x
    _, _, _, _, info = env.step(action)

    assert info["legal_action"] is True
    assert int(info["own_active_units"]) == 0


def test_existing_friendly_unit_deals_ongoing_damage_on_noop() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    start_enemy_hp = float(env.enemy_king_hp)
    env.own_units = [
        ActiveUnit(
            x=env.cfg.grid_w // 2,
            y=env.river_top_y,
            hp=50.0,
            dps=20.0,
            ttl=4,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
            attack_range=2,
            attack_cooldown_steps=2,
            cooldown_remaining=0,
            hit_damage=12.0,
        )
    ]
    _, _, _, _, info = env.step(env.noop_action)
    assert float(info["ongoing_damage_to_enemy"]) > 0.0
    assert float(env.enemy_king_hp) < start_enemy_hp


def test_tower_attacks_respect_cooldown_steps() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0, attack_cooldown_steps=3))
    env.reset(seed=0)
    env.own_units = [
        ActiveUnit(
            x=env.cfg.grid_w // 2,
            y=env.river_top_y,
            hp=40.0,
            dps=10.0,
            ttl=8,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
            attack_range=2,
            attack_cooldown_steps=3,
            cooldown_remaining=0,
            hit_damage=20.0,
        )
    ]
    hp0 = float(env.enemy_king_hp)
    env.step(env.noop_action)
    hp1 = float(env.enemy_king_hp)
    env.step(env.noop_action)
    hp2 = float(env.enemy_king_hp)
    assert hp1 < hp0
    assert hp2 == hp1


def test_projectile_mode_delays_ranged_tower_damage() -> None:
    env = CrLikeSimEnv(
        config=SimConfig(
            enemy_spawn_chance=0.0,
            projectile_travel_enabled=True,
            projectile_speed_cells_per_step=1.0,
            attack_cooldown_steps=2,
        )
    )
    env.reset(seed=0)
    env.own_units = [
        ActiveUnit(
            x=env.cfg.grid_w // 2,
            y=env.river_top_y,
            hp=40.0,
            dps=10.0,
            ttl=8,
            card_type="building",
            target_type="ground",
            can_hit_air=False,
            is_air=False,
            is_enemy=False,
            attack_range=4,
            attack_cooldown_steps=2,
            cooldown_remaining=0,
            hit_damage=20.0,
            splash_radius=0.0,
        )
    ]
    hp0 = float(env.enemy_king_hp)
    _, _, _, _, info1 = env.step(env.noop_action)
    hp1 = float(env.enemy_king_hp)
    # First step should queue projectile but not impact yet.
    assert int(info1["pending_projectiles"]) >= 1
    assert hp1 == hp0
    hp_after = hp1
    for _ in range(10):
        env.step(env.noop_action)
        hp_after = float(env.enemy_king_hp)
        if hp_after < hp1:
            break
    assert hp_after < hp1


def test_friendly_troop_advances_toward_enemy_side() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    start_y = min(env.cfg.grid_h - 1, env.deploy_min_y + 3)
    env.own_units = [
        ActiveUnit(
            x=env.cfg.grid_w // 2,
            y=start_y,
            hp=40.0,
            dps=10.0,
            ttl=4,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
        )
    ]
    env.step(env.noop_action)
    assert env.own_units
    assert env.own_units[0].y == start_y - 1


def test_target_priority_hits_closest_enemy_in_range() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    env.own_units = [
        ActiveUnit(
            x=4,
            y=8,
            hp=60.0,
            dps=20.0,
            ttl=5,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
        )
    ]
    close = ActiveUnit(
        x=4,
        y=7,
        hp=50.0,
        dps=5.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=True,
    )
    far = ActiveUnit(
        x=0,
        y=0,
        hp=50.0,
        dps=5.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=True,
    )
    env.enemy_units = [close, far]
    close_hp_before = close.hp
    far_hp_before = far.hp
    env.step(env.noop_action)
    close_loss = close_hp_before - close.hp
    far_loss = far_hp_before - far.hp
    assert close_loss > far_loss


def test_friendly_troop_moves_toward_bridge_before_river_crossing() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    start_y = env.river_bottom_y + 1
    start_x = 0  # not a bridge lane in default config
    env.own_units = [
        ActiveUnit(
            x=start_x,
            y=start_y,
            hp=40.0,
            dps=8.0,
            ttl=6,
            card_type="troop",
            target_type="ground",
            can_hit_air=False,
            is_air=False,
            is_enemy=False,
        )
    ]
    env.step(env.noop_action)
    assert env.own_units
    unit = env.own_units[0]
    # Should sidestep toward nearest bridge instead of entering river row.
    assert unit.y == start_y
    assert unit.x != start_x


def test_friendly_troop_can_enter_river_on_bridge_lane() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    start_y = env.river_bottom_y + 1
    bridge_x = env._nearest_bridge_x(0)
    env.own_units = [
        ActiveUnit(
            x=bridge_x,
            y=start_y,
            hp=40.0,
            dps=8.0,
            ttl=6,
            card_type="troop",
            target_type="ground",
            can_hit_air=False,
            is_air=False,
            is_enemy=False,
        )
    ]
    env.step(env.noop_action)
    assert env.own_units
    unit = env.own_units[0]
    assert unit.y == env.river_bottom_y


def test_state_snapshot_and_observation_builder_roundtrip() -> None:
    env = CrLikeSimEnv()
    env.reset(seed=0)
    snap = env.get_state_snapshot()
    obs = env.build_observation_from_state(snap)
    assert set(obs.keys()) == {"global", "hand_ids", "hand_costs"}
    assert obs["global"].shape == (8,)
    assert obs["hand_ids"].shape == (env.cfg.hand_size,)
    assert obs["hand_costs"].shape == (env.cfg.hand_size,)


def test_optional_low_res_unit_density_observation() -> None:
    cfg = SimConfig(observe_unit_density=True, obs_grid_w=5, obs_grid_h=6, enemy_spawn_chance=0.0)
    env = CrLikeSimEnv(config=cfg)
    obs, _ = env.reset(seed=0)
    assert "unit_density" in obs
    assert obs["unit_density"].shape == (2, 6, 5)
    flat = flatten_observation(obs)
    assert flat.shape[0] == 8 + env.cfg.hand_size + env.cfg.hand_size + (2 * 6 * 5)


def test_unit_does_not_move_into_occupied_forward_cell() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0, max_units_per_cell=1))
    env.reset(seed=0)
    mover_start_y = min(env.cfg.grid_h - 1, env.deploy_min_y + 3)
    blocker_y = mover_start_y - 1
    x = env.cfg.grid_w // 2
    mover = ActiveUnit(
        x=x,
        y=mover_start_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    blocker = ActiveUnit(
        x=x,
        y=blocker_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="building",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    env.own_units = [mover, blocker]
    env.step(env.noop_action)
    assert env.own_units
    moved = next(u for u in env.own_units if u is mover)
    assert moved.y == mover_start_y


def test_unit_does_not_sidestep_into_occupied_bridge_cell() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0, max_units_per_cell=1))
    env.reset(seed=0)
    start_y = env.river_bottom_y + 1
    start_x = 0
    sidestep_target_x = 1
    mover = ActiveUnit(
        x=start_x,
        y=start_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    blocker = ActiveUnit(
        x=sidestep_target_x,
        y=start_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    env.own_units = [mover, blocker]
    env.step(env.noop_action)
    assert env.own_units
    moved = next(u for u in env.own_units if u is mover)
    assert moved.x == start_x
    assert moved.y == start_y


def test_left_lane_unit_pressures_left_princess_more() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    left_before = float(env.enemy_princess_hps[0])
    right_before = float(env.enemy_princess_hps[1])
    env.own_units = [
        ActiveUnit(
            x=0,
            y=env.river_top_y,
            hp=50.0,
            dps=20.0,
            ttl=4,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
            attack_range=2,
            attack_cooldown_steps=2,
            cooldown_remaining=0,
            hit_damage=14.0,
        )
    ]
    env.step(env.noop_action)
    left_loss = left_before - float(env.enemy_princess_hps[0])
    right_loss = right_before - float(env.enemy_princess_hps[1])
    assert left_loss > 0.0
    assert right_loss == 0.0


def test_tower_pressure_retargets_when_preferred_princess_is_destroyed() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    env.enemy_princess_hps[0] = 0.0
    right_before = float(env.enemy_princess_hps[1])
    env.own_units = [
        ActiveUnit(
            x=0,
            y=env.river_top_y,
            hp=50.0,
            dps=20.0,
            ttl=4,
            card_type="troop",
            target_type="any",
            can_hit_air=True,
            is_air=False,
            is_enemy=False,
            attack_range=2,
            attack_cooldown_steps=2,
            cooldown_remaining=0,
            hit_damage=14.0,
        )
    ]
    env.step(env.noop_action)
    assert float(env.enemy_princess_hps[1]) < right_before


def test_enemy_side_pathing_drifts_toward_lane_objective() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    unit = ActiveUnit(
        x=env.cfg.grid_w // 2,
        y=env.river_top_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    env.own_units = [unit]
    env.step(env.noop_action)
    assert env.own_units
    moved = env.own_units[0]
    assert moved.x == (env.cfg.grid_w // 2) + 1


def test_enemy_side_pathing_switches_lane_when_preferred_princess_is_down() -> None:
    env = CrLikeSimEnv(config=SimConfig(enemy_spawn_chance=0.0))
    env.reset(seed=0)
    env.enemy_princess_hps[1] = 0.0
    unit = ActiveUnit(
        x=env.cfg.grid_w - 2,
        y=env.river_top_y,
        hp=40.0,
        dps=8.0,
        ttl=5,
        card_type="troop",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
    )
    env.own_units = [unit]
    x_before = unit.x
    env.step(env.noop_action)
    assert env.own_units
    moved = env.own_units[0]
    assert moved.x < x_before


def test_unit_profile_has_archetype_specific_projectile_speed() -> None:
    env = CrLikeSimEnv()
    building = env.get_card_meta(env.cfg.spell_card_count)
    troop_any = env.get_card_meta(env.cfg.spell_card_count + env.cfg.building_card_count + 1)

    _, _, _, _, _, _, _, building_speed = env._unit_profile(building, float(building.elixir_cost))
    _, _, _, _, _, _, _, troop_speed = env._unit_profile(troop_any, float(troop_any.elixir_cost))
    assert troop_speed > building_speed


def test_faster_projectile_hits_before_slower_projectile() -> None:
    env = CrLikeSimEnv(
        config=SimConfig(
            enemy_spawn_chance=0.0,
            projectile_travel_enabled=True,
            projectile_speed_cells_per_step=2.0,
            attack_cooldown_steps=50,
        )
    )
    env.reset(seed=0)

    slow = ActiveUnit(
        x=0,
        y=env.river_top_y,
        hp=40.0,
        dps=10.0,
        ttl=8,
        card_type="building",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
        attack_range=4,
        attack_cooldown_steps=50,
        cooldown_remaining=0,
        hit_damage=20.0,
        splash_radius=0.0,
        projectile_speed_cells_per_step=1.0,
    )
    fast = ActiveUnit(
        x=env.cfg.grid_w - 1,
        y=env.river_top_y,
        hp=40.0,
        dps=10.0,
        ttl=8,
        card_type="building",
        target_type="ground",
        can_hit_air=False,
        is_air=False,
        is_enemy=False,
        attack_range=4,
        attack_cooldown_steps=50,
        cooldown_remaining=0,
        hit_damage=20.0,
        splash_radius=0.0,
        projectile_speed_cells_per_step=4.0,
    )
    env.own_units = [slow, fast]
    hp0 = float(env.enemy_king_hp)
    env.step(env.noop_action)  # queue projectiles
    hp1 = float(env.enemy_king_hp)
    assert hp1 == hp0

    # Fast projectile should land first (within a few steps), while slow one lags.
    hp_after = hp1
    for _ in range(4):
        env.step(env.noop_action)
        hp_after = float(env.enemy_king_hp)
        if hp_after < hp1:
            break
    assert hp_after < hp1
