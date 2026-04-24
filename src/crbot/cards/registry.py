from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    archetypes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] | None = None


def default_registry_path() -> Path:
    # .../src/crbot/cards/registry.py -> repo root -> configs/cards_registry.yaml
    return Path(__file__).resolve().parents[3] / "configs" / "cards_registry.yaml"


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError(f"Expected string or list for archetypes/inherits, got: {type(raw).__name__}")


def _resolve_archetype(
    name: str,
    archetypes: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    stack: set[str],
) -> dict[str, Any]:
    n = str(name).strip()
    if n in cache:
        return cache[n]
    if n in stack:
        raise ValueError(f"Archetype inheritance cycle detected at: {n}")
    if n not in archetypes:
        raise ValueError(f"Unknown archetype: {n}")
    stack.add(n)
    raw = dict(archetypes[n])
    parents = _normalize_names(raw.pop("inherits", None))
    merged: dict[str, Any] = {}
    for p in parents:
        merged = _merge_dicts(merged, _resolve_archetype(p, archetypes, cache, stack))
    merged = _merge_dicts(merged, raw)
    stack.remove(n)
    cache[n] = merged
    return merged


def _validate_entry(raw: dict[str, Any]) -> CardRegistryEntry:
    card_type = str(raw["card_type"]).strip().lower()
    target_type = str(raw["target_type"]).strip().lower()
    if card_type not in {"spell", "building", "troop"}:
        raise ValueError(f"Invalid card_type: {card_type}")
    if target_type not in {"ground", "air", "any", "area"}:
        raise ValueError(f"Invalid target_type: {target_type}")
    aliases_raw = raw.get("aliases", [])
    aliases = tuple(str(x).strip().lower() for x in aliases_raw if str(x).strip())
    archetypes = tuple(str(x).strip() for x in _normalize_names(raw.get("archetypes")) if str(x).strip())
    tags_raw = raw.get("tags", [])
    tags = tuple(str(x).strip().lower() for x in tags_raw if str(x).strip())
    extra = None
    if isinstance(raw.get("extra"), dict):
        extra = dict(raw["extra"])
    return CardRegistryEntry(
        card_id=int(raw["card_id"]),
        name=str(raw["name"]).strip(),
        aliases=aliases,
        elixir_cost=int(raw["elixir_cost"]),
        card_type=card_type,
        target_type=target_type,
        can_hit_air=bool(raw["can_hit_air"]),
        archetypes=archetypes,
        tags=tags,
        extra=extra,
    )


def load_card_registry(path: str | Path | None = None) -> list[CardRegistryEntry]:
    p = Path(path) if path is not None else default_registry_path()
    if not p.exists():
        raise FileNotFoundError(p)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cards"), list):
        raise ValueError(f"Invalid card registry format at {p}")

    archetypes_raw = raw.get("archetypes", {})
    if archetypes_raw is None:
        archetypes_raw = {}
    if not isinstance(archetypes_raw, dict):
        raise ValueError("archetypes must be a mapping from name -> defaults.")
    archetypes: dict[str, dict[str, Any]] = {}
    for k, v in archetypes_raw.items():
        if not isinstance(v, dict):
            raise ValueError(f"Archetype {k} must map to an object.")
        archetypes[str(k)] = dict(v)
    resolved_cache: dict[str, dict[str, Any]] = {}

    merged_cards: list[dict[str, Any]] = []
    for card_raw in raw["cards"]:
        if not isinstance(card_raw, dict):
            raise ValueError("Each card entry must be an object.")
        card = dict(card_raw)
        archetype_names = _normalize_names(card.get("archetypes", card.get("inherits")))
        merged: dict[str, Any] = {}
        for a in archetype_names:
            merged = _merge_dicts(merged, _resolve_archetype(a, archetypes, resolved_cache, set()))
        merged = _merge_dicts(merged, card)
        merged["archetypes"] = archetype_names
        if "inherits" in merged:
            del merged["inherits"]
        merged_cards.append(merged)

    entries = [_validate_entry(x) for x in merged_cards]
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
