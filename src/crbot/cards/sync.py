from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OfficialCard:
    api_id: int
    name: str
    max_level: int | None = None
    icon_url: str | None = None


def normalize_card_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def parse_official_cards_payload(payload: dict[str, Any]) -> list[OfficialCard]:
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Official cards payload must include list field 'items'.")
    out: list[OfficialCard] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "id" not in it or "name" not in it:
            continue
        icon = None
        icon_urls = it.get("iconUrls")
        if isinstance(icon_urls, dict):
            icon = icon_urls.get("medium") or icon_urls.get("evolutionMedium")
        out.append(
            OfficialCard(
                api_id=int(it["id"]),
                name=str(it["name"]),
                max_level=None if it.get("maxLevel") is None else int(it.get("maxLevel")),
                icon_url=None if icon is None else str(icon),
            )
        )
    return out


def merge_registry_with_official_cards(registry_raw: dict[str, Any], official_cards: list[OfficialCard]) -> tuple[dict[str, Any], dict[str, int]]:
    cards = registry_raw.get("cards")
    if not isinstance(cards, list):
        raise ValueError("Registry must contain list field 'cards'.")
    # Build normalized index for existing cards and aliases.
    name_to_idx: dict[str, int] = {}
    max_card_id = -1
    for i, c in enumerate(cards):
        if not isinstance(c, dict):
            continue
        max_card_id = max(max_card_id, int(c.get("card_id", -1)))
        name_to_idx[normalize_card_name(c.get("name", ""))] = i
        for a in c.get("aliases", []):
            name_to_idx[normalize_card_name(a)] = i

    added = 0
    matched = 0
    updated = 0
    for oc in official_cards:
        key = normalize_card_name(oc.name)
        idx = name_to_idx.get(key)
        if idx is not None:
            matched += 1
            entry = cards[idx]
            if entry.get("official_api_id") is None:
                entry["official_api_id"] = int(oc.api_id)
                updated += 1
            extra = entry.get("extra")
            if not isinstance(extra, dict):
                extra = {}
            if oc.max_level is not None:
                extra["api_max_level"] = int(oc.max_level)
            if oc.icon_url:
                extra["api_icon_url"] = str(oc.icon_url)
            if extra:
                entry["extra"] = extra
            continue

        max_card_id += 1
        new_entry: dict[str, Any] = {
            "card_id": int(max_card_id),
            "official_api_id": int(oc.api_id),
            "archetypes": ["troop_ground"],
            "name": str(oc.name),
            "aliases": [str(oc.name).strip().lower()],
            "elixir_cost": 4,
            "tags": ["api_stub", "needs_review"],
            "extra": {},
        }
        if oc.max_level is not None:
            new_entry["extra"]["api_max_level"] = int(oc.max_level)
        if oc.icon_url:
            new_entry["extra"]["api_icon_url"] = str(oc.icon_url)
        cards.append(new_entry)
        name_to_idx[key] = len(cards) - 1
        added += 1

    stats = {
        "official_total": len(official_cards),
        "matched_existing": matched,
        "updated_existing": updated,
        "added_stubs": added,
        "registry_total_after": len(cards),
    }
    return registry_raw, stats
