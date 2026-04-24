from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from crbot.cards import (
    apply_bulk_registry_review_edits,
    default_registry_path,
    list_registry_cards_for_review,
    summarize_registry_review_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and bulk-edit card registry stubs.")
    parser.add_argument("--registry", type=str, default=str(default_registry_path()))
    parser.add_argument("--tag", type=str, default="needs_review", help="Filter tag for listing/editing.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print for --list.")
    parser.add_argument("--summary", action="store_true", help="Print registry review summary.")
    parser.add_argument("--list", action="store_true", help="List cards matching --tag.")
    parser.add_argument("--set-archetype", type=str, default="", help="Set archetype for all cards matching --tag.")
    parser.add_argument("--add-tag", action="append", default=[], help="Tag to add to matching cards.")
    parser.add_argument("--remove-tag", action="append", default=[], help="Tag to remove from matching cards.")
    parser.add_argument("--apply", action="store_true", help="Write changes to registry.")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    if args.summary:
        summary = summarize_registry_review_state(raw)
        print("[summary]")
        for k, v in summary.items():
            print(f"- {k}: {v}")

    if args.list:
        rows = list_registry_cards_for_review(raw, tag=args.tag, limit=args.limit)
        print(f"[list tag={args.tag} count={len(rows)}]")
        for r in rows:
            print(
                f"- card_id={r['card_id']} name={r['name']} api_id={r['official_api_id']} "
                f"archetypes={r['archetypes']} tags={r['tags']}"
            )

    wants_edit = bool(args.set_archetype.strip() or args.add_tag or args.remove_tag)
    if wants_edit:
        edited, stats = apply_bulk_registry_review_edits(
            raw,
            where_tag=args.tag,
            set_archetype=args.set_archetype.strip() or None,
            add_tags=args.add_tag,
            remove_tags=args.remove_tag,
        )
        print("[edit-summary]")
        for k, v in stats.items():
            print(f"- {k}: {v}")
        if args.apply:
            registry_path.write_text(yaml.safe_dump(edited, sort_keys=False, allow_unicode=False), encoding="utf-8")
            print(f"[done] updated registry: {registry_path.resolve()}")
        else:
            print("[dry-run] no files changed. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
