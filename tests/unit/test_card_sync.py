from crbot.cards.sync import OfficialCard, merge_registry_with_official_cards, parse_official_cards_payload


def test_parse_official_cards_payload_extracts_core_fields() -> None:
    payload = {
        "items": [
            {
                "id": 26000000,
                "name": "Knight",
                "maxLevel": 14,
                "iconUrls": {"medium": "https://example.com/knight.png"},
            }
        ]
    }
    out = parse_official_cards_payload(payload)
    assert len(out) == 1
    assert out[0].api_id == 26000000
    assert out[0].name == "Knight"
    assert out[0].max_level == 14
    assert out[0].icon_url == "https://example.com/knight.png"


def test_merge_registry_matches_existing_and_sets_official_id() -> None:
    registry = {
        "cards": [
            {
                "card_id": 0,
                "name": "Knight",
                "aliases": ["knight"],
                "archetypes": ["troop_ground"],
                "elixir_cost": 3,
            }
        ]
    }
    official = [OfficialCard(api_id=123, name="Knight", max_level=14, icon_url="u")]
    merged, stats = merge_registry_with_official_cards(registry, official)
    c = merged["cards"][0]
    assert c["official_api_id"] == 123
    assert c["extra"]["api_max_level"] == 14
    assert stats["updated_existing"] == 1
    assert stats["added_stubs"] == 0


def test_merge_registry_adds_stub_for_new_card_with_contiguous_id() -> None:
    registry = {
        "cards": [
            {"card_id": 0, "name": "Knight", "aliases": ["knight"], "archetypes": ["troop_ground"], "elixir_cost": 3}
        ]
    }
    official = [OfficialCard(api_id=999, name="Mega Knight", max_level=14, icon_url=None)]
    merged, stats = merge_registry_with_official_cards(registry, official)
    assert len(merged["cards"]) == 2
    new_card = merged["cards"][1]
    assert new_card["card_id"] == 1
    assert new_card["official_api_id"] == 999
    assert "needs_review" in new_card["tags"]
    assert stats["added_stubs"] == 1
