from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CardRegistryEntry:
    card_id: int
    name: str
    aliases: tuple[str, ...]
    elixir_cost: int
    card_type: str  # "spell" | "building" | "troop"
    target_type: str  # "ground" | "air" | "any" | "area"
    can_hit_air: bool


def default_registry_path() -> Path:
    # .../src/crbot/cards/registry.py -> repo root -> configs/cards_registry.yaml
    return Path(__file__).resolve().parents[3] / "configs" / "cards_registry.yaml"


def _validate_entry(raw: dict) -> CardRegistryEntry:
    card_type = str(raw["card_type"]).strip().lower()
    target_type = str(raw["target_type"]).strip().lower()
    if card_type not in {"spell", "building", "troop"}:
        raise ValueError(f"Invalid card_type: {card_type}")
    if target_type not in {"ground", "air", "any", "area"}:
        raise ValueError(f"Invalid target_type: {target_type}")
    aliases_raw = raw.get("aliases", [])
    aliases = tuple(str(x).strip().lower() for x in aliases_raw if str(x).strip())
    return CardRegistryEntry(
        card_id=int(raw["card_id"]),
        name=str(raw["name"]).strip(),
        aliases=aliases,
        elixir_cost=int(raw["elixir_cost"]),
        card_type=card_type,
        target_type=target_type,
        can_hit_air=bool(raw["can_hit_air"]),
    )


def load_card_registry(path: str | Path | None = None) -> list[CardRegistryEntry]:
    p = Path(path) if path is not None else default_registry_path()
    if not p.exists():
        raise FileNotFoundError(p)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cards"), list):
        raise ValueError(f"Invalid card registry format at {p}")
    entries = [_validate_entry(x) for x in raw["cards"]]
    if not entries:
        raise ValueError("Card registry must include at least one card.")
    ids = [e.card_id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate card_id in card registry.")
    entries = sorted(entries, key=lambda e: e.card_id)
    expected_ids = list(range(entries[-1].card_id + 1))
    if [e.card_id for e in entries] != expected_ids:
        raise ValueError("Card registry card_id values must be contiguous starting at 0.")
    return entries


def build_card_catalog_from_registry(path: str | Path | None = None) -> dict[int, CardRegistryEntry]:
    entries = load_card_registry(path=path)
    return {e.card_id: e for e in entries}
