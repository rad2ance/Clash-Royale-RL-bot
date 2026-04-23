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
    assert bool(mask.all()) is True


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
