import numpy as np

from crbot.sim import CrLikeSimEnv, flatten_observation


def test_env_reset_and_step_shapes() -> None:
    env = CrLikeSimEnv()
    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
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

    next_flat = flatten_observation(next_obs)
    assert next_flat.shape == flat.shape
    assert np.isfinite(next_flat).all()

