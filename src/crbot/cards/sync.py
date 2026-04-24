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


def summarize_registry_review_state(registry_raw: dict[str, Any]) -> dict[str, int]:
    cards = registry_raw.get("cards", [])
    if not isinstance(cards, list):
        raise ValueError("Registry must contain list field 'cards'.")
    total = 0
    needs_review = 0
    api_stub = 0
    with_official_id = 0
    for c in cards:
        if not isinstance(c, dict):
            continue
        total += 1
        tags = {str(x).strip().lower() for x in c.get("tags", [])}
        if "needs_review" in tags:
            needs_review += 1
        if "api_stub" in tags:
            api_stub += 1
        if c.get("official_api_id") is not None:
            with_official_id += 1
    return {
        "total_cards": total,
        "needs_review": needs_review,
        "api_stubs": api_stub,
        "with_official_api_id": with_official_id,
    }


def list_registry_cards_for_review(registry_raw: dict[str, Any], *, tag: str = "needs_review", limit: int = 100) -> list[dict[str, Any]]:
    cards = registry_raw.get("cards", [])
    if not isinstance(cards, list):
        raise ValueError("Registry must contain list field 'cards'.")
    t = str(tag).strip().lower()
    out: list[dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        tags = {str(x).strip().lower() for x in c.get("tags", [])}
        if t and t not in tags:
            continue
        out.append(
            {
                "card_id": int(c.get("card_id", -1)),
                "name": str(c.get("name", "")),
                "official_api_id": c.get("official_api_id"),
                "archetypes": list(c.get("archetypes", [])),
                "elixir_cost": c.get("elixir_cost"),
                "tags": list(c.get("tags", [])),
            }
        )
        if len(out) >= max(0, int(limit)):
            break
    return out


def apply_bulk_registry_review_edits(
    registry_raw: dict[str, Any],
    *,
    where_tag: str = "needs_review",
    set_archetype: str | None = None,
    add_tags: list[str] | None = None,
    remove_tags: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    cards = registry_raw.get("cards")
    if not isinstance(cards, list):
        raise ValueError("Registry must contain list field 'cards'.")
    target_tag = str(where_tag).strip().lower()
    add = [str(x).strip().lower() for x in (add_tags or []) if str(x).strip()]
    remove = [str(x).strip().lower() for x in (remove_tags or []) if str(x).strip()]

    touched = 0
    for c in cards:
        if not isinstance(c, dict):
            continue
        tags = [str(x).strip().lower() for x in c.get("tags", []) if str(x).strip()]
        tag_set = set(tags)
        if target_tag and target_tag not in tag_set:
            continue
        touched += 1
        if set_archetype:
            c["archetypes"] = [str(set_archetype).strip()]
        for t in add:
            tag_set.add(t)
        for t in remove:
            if t in tag_set:
                tag_set.remove(t)
        c["tags"] = sorted(tag_set)

    stats = {"touched_cards": touched}
    return registry_raw, stats
