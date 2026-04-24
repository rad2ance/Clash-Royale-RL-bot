from crbot.cards.sync import (
    apply_bulk_registry_review_edits,
    list_registry_cards_for_review,
    summarize_registry_review_state,
)


def _registry() -> dict:
    return {
        "cards": [
            {
                "card_id": 0,
                "name": "Knight",
                "archetypes": ["troop_ground"],
                "tags": ["api_stub", "needs_review"],
                "official_api_id": 1,
            },
            {
                "card_id": 1,
                "name": "Arrows",
                "archetypes": ["spell_area"],
                "tags": ["spell"],
                "official_api_id": 2,
            },
        ]
    }


def test_summarize_registry_review_state_counts_tags() -> None:
    s = summarize_registry_review_state(_registry())
    assert s["total_cards"] == 2
    assert s["needs_review"] == 1
    assert s["api_stubs"] == 1
    assert s["with_official_api_id"] == 2


def test_list_registry_cards_for_review_filters_by_tag() -> None:
    rows = list_registry_cards_for_review(_registry(), tag="needs_review", limit=10)
    assert len(rows) == 1
    assert rows[0]["name"] == "Knight"


def test_apply_bulk_registry_review_edits_updates_only_matching_tag() -> None:
    raw, stats = apply_bulk_registry_review_edits(
        _registry(),
        where_tag="needs_review",
        set_archetype="troop_any",
        add_tags=["reviewed"],
        remove_tags=["needs_review"],
    )
    assert stats["touched_cards"] == 1
    c0 = raw["cards"][0]
    c1 = raw["cards"][1]
    assert c0["archetypes"] == ["troop_any"]
    assert "reviewed" in c0["tags"]
    assert "needs_review" not in c0["tags"]
    assert c1["archetypes"] == ["spell_area"]
