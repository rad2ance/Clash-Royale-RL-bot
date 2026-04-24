from pathlib import Path

import pytest

from crbot.cards import default_registry_path, load_card_registry
from crbot.sim import CrLikeSimEnv, SimConfig


def test_default_registry_exists_and_is_contiguous() -> None:
    p = default_registry_path()
    assert p.exists()
    entries = load_card_registry(p)
    assert len(entries) > 0
    assert [e.card_id for e in entries] == list(range(len(entries)))


def test_sim_uses_registry_names_for_default_card_catalog() -> None:
    env = CrLikeSimEnv(config=SimConfig())
    assert env.get_card_meta(0).name == "Arrows"
    assert env.get_card_meta(2).name == "Fireball"


def test_sim_raises_on_registry_size_mismatch(tmp_path: Path) -> None:
    # Build a tiny invalid-for-config registry to ensure mismatch check protects setup.
    path = tmp_path / "cards_registry.yaml"
    path.write_text(
        "\n".join(
            [
                "cards:",
                "  - card_id: 0",
                "    name: Test",
                "    aliases: []",
                "    elixir_cost: 2",
                "    card_type: troop",
                "    target_type: ground",
                "    can_hit_air: false",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        CrLikeSimEnv(config=SimConfig(n_cards=12, card_registry_path=str(path)))
