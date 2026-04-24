from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-apply safe archetype suggestions from card review backlog."
    )
    parser.add_argument("--registry", type=str, default="configs/cards_registry.yaml")
    parser.add_argument("--backlog", type=str, default="data/cards/review_backlog.jsonl")
    parser.add_argument(
        "--only-keyword",
        action="store_true",
        default=True,
        help="Only apply rows where suggestion_reason starts with 'keyword:'.",
    )
    parser.add_argument(
        "--allow-default",
        action="store_true",
        help="Also apply default suggestions (less safe).",
    )
    parser.add_argument(
        "--allowed-archetypes",
        nargs="+",
        default=["spell_area", "building_ground"],
        help="Archetypes allowed for auto-apply.",
    )
    parser.add_argument(
        "--remove-tag",
        action="append",
        default=["needs_review"],
        help="Tag(s) to remove on updated cards.",
    )
    parser.add_argument(
        "--add-tag",
        action="append",
        default=["reviewed_auto"],
        help="Tag(s) to add on updated cards.",
    )
    parser.add_argument("--apply", action="store_true", help="Write changes to registry.")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    backlog_path = Path(args.backlog)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    if not backlog_path.exists():
        raise FileNotFoundError(backlog_path)

    reg = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    cards = reg.get("cards", [])
    if not isinstance(cards, list):
        raise ValueError("Invalid registry format: cards must be a list.")
    card_by_id = {
        int(c.get("card_id")): c
        for c in cards
        if isinstance(c, dict) and c.get("card_id") is not None
    }
    backlog_rows = _load_jsonl(backlog_path)

    allowed_archetypes = {str(x).strip() for x in args.allowed_archetypes if str(x).strip()}
    remove_tags = {str(x).strip().lower() for x in args.remove_tag if str(x).strip()}
    add_tags = {str(x).strip().lower() for x in args.add_tag if str(x).strip()}

    touched = 0
    for row in backlog_rows:
        cid = int(row.get("card_id", -1))
        suggested = str(row.get("suggested_archetype", "")).strip()
        reason = str(row.get("suggestion_reason", "")).strip().lower()
        if cid not in card_by_id:
            continue
        if suggested not in allowed_archetypes:
            continue
        if args.only_keyword and (not reason.startswith("keyword:")) and (not args.allow_default):
            continue
        card = card_by_id[cid]
        card["archetypes"] = [suggested]
        tags = {str(t).strip().lower() for t in card.get("tags", []) if str(t).strip()}
        for t in remove_tags:
            tags.discard(t)
        tags |= add_tags
        card["tags"] = sorted(tags)
        touched += 1

    print("[auto-review]")
    print(f"- backlog_rows: {len(backlog_rows)}")
    print(f"- touched_cards: {touched}")
    print(f"- allowed_archetypes: {sorted(allowed_archetypes)}")
    print(f"- mode: {'apply' if args.apply else 'dry-run'}")

    if args.apply:
        registry_path.write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=False), encoding="utf-8")
        print(f"[done] updated registry: {registry_path.resolve()}")
    else:
        print("[dry-run] no files changed. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
