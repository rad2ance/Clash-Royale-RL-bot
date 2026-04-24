from crbot.cards.sync import (
    build_registry_review_backlog,
    compute_review_priority,
    guess_archetype_for_card_name,
)


def test_guess_archetype_for_card_name_spell_keyword() -> None:
    arch, reason = guess_archetype_for_card_name("Fireball")
    assert arch == "spell_area"
    assert "keyword" in reason


def test_guess_archetype_for_card_name_building_keyword() -> None:
    arch, _ = guess_archetype_for_card_name("Bomb Tower")
    assert arch == "building_ground"


def test_guess_archetype_for_card_name_air_keyword() -> None:
    arch, _ = guess_archetype_for_card_name("Inferno Dragon")
    assert arch == "troop_any"


def test_compute_review_priority_prefers_needs_review_and_stub() -> None:
    low = {"tags": [], "official_api_id": 1}
    high = {"tags": ["needs_review", "api_stub"], "official_api_id": None, "extra": {"api_icon_url": "x"}}
    assert compute_review_priority(high) > compute_review_priority(low)


def test_build_registry_review_backlog_filters_and_sorts() -> None:
    raw = {
        "cards": [
            {"card_id": 0, "name": "Knight", "tags": [], "archetypes": ["troop_ground"], "official_api_id": 11},
            {
                "card_id": 1,
                "name": "Inferno Dragon",
                "tags": ["needs_review", "api_stub"],
                "archetypes": ["troop_ground"],
                "official_api_id": None,
            },
            {
                "card_id": 2,
                "name": "Zap",
                "tags": ["needs_review"],
                "archetypes": ["troop_ground"],
                "official_api_id": 99,
            },
        ]
    }
    backlog = build_registry_review_backlog(raw, limit=10, only_needs_review=True)
    assert [x["card_id"] for x in backlog] == [1, 2]
    assert backlog[0]["suggested_archetype"] == "troop_any"
