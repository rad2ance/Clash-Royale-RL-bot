from pathlib import Path

import numpy as np

from crbot.data import EpisodeBatch, load_episode, save_episode


def test_save_load_roundtrip(tmp_path: Path) -> None:
    ep = EpisodeBatch(
        observations=np.random.randn(10, 16).astype(np.float32),
        actions=np.random.randint(0, 9, size=(10,), dtype=np.int64),
        rewards=np.random.randn(10).astype(np.float32),
        dones=np.zeros(10, dtype=bool),
    )
    out = tmp_path / "ep.npz"
    save_episode(out, ep)
    loaded = load_episode(out)

    assert loaded.observations.shape == (10, 16)
    assert loaded.actions.shape == (10,)
    assert loaded.rewards.shape == (10,)
    assert loaded.dones.shape == (10,)

