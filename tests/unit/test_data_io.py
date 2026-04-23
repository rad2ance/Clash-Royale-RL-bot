from pathlib import Path

import numpy as np

from crbot.data import EpisodeBatch, load_episode, save_episode


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ep = EpisodeBatch(
        observations=np.random.randn(10, 16).astype(np.float32),
        actions=np.random.randint(0, 9, size=(10,), dtype=np.int64),
        rewards=np.random.randn(10).astype(np.float32),
        dones=np.zeros(10, dtype=bool),
        action_masks=np.random.randint(0, 2, size=(10, 32), dtype=np.int8).astype(bool),
    )
    out = tmp_path / "ep.npz"
    save_episode(out, ep)
    loaded = load_episode(out)

    assert loaded.observations.shape == (10, 16)
    assert loaded.actions.shape == (10,)
    assert loaded.rewards.shape == (10,)
    assert loaded.dones.shape == (10,)
    assert loaded.action_masks is not None
    assert loaded.action_masks.shape == (10, 32)


def test_load_episode_backward_compatible_without_masks(tmp_path: Path) -> None:
    out = tmp_path / "legacy_ep.npz"
    np.savez_compressed(
        out,
        observations=np.random.randn(6, 4).astype(np.float32),
        actions=np.random.randint(0, 5, size=(6,), dtype=np.int64),
        rewards=np.random.randn(6).astype(np.float32),
        dones=np.zeros(6, dtype=bool),
    )
    loaded = load_episode(out)
    assert loaded.action_masks is None
