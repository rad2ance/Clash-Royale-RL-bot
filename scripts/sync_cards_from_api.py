from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import error, request

import yaml

from crbot.cards import default_registry_path
from crbot.cards.sync import merge_registry_with_official_cards, parse_official_cards_payload


def _fetch_official_cards(api_token: str, base_url: str) -> dict:
    url = base_url.rstrip("/") + "/v1/cards"
    req = request.Request(url)
    req.add_header("Authorization", f"Bearer {api_token}")
    req.add_header("Accept", "application/json")
    try:
        with request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while fetching cards from {url}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to connect to {url}: {exc}") from exc
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge official Clash Royale card list into local cards registry.")
    parser.add_argument("--registry", type=str, default=str(default_registry_path()))
    parser.add_argument("--from-json", type=str, default="", help="Path to saved official /v1/cards JSON payload.")
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save fetched/loaded payload for reuse.",
    )
    parser.add_argument("--base-url", type=str, default="https://api.clashroyale.com")
    parser.add_argument(
        "--token-env",
        type=str,
        default="CR_API_TOKEN",
        help="Env var name containing Clash Royale API token (used when --from-json is not set).",
    )
    parser.add_argument("--apply", action="store_true", help="Write merged registry to disk.")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)

    if args.from_json:
        json_path = Path(args.from_json)
        if not json_path.exists():
            raise RuntimeError(
                f"--from-json file not found: {json_path}\n"
                "Use a valid payload path, or fetch directly from API by setting CR_API_TOKEN and omitting --from-json."
            )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source = f"json:{json_path.resolve()}"
    else:
        token = os.getenv(args.token_env, "")
        if not token:
            raise RuntimeError(
                f"Missing API token env var {args.token_env}. "
                f"Set it or pass --from-json with a saved payload."
            )
        payload = _fetch_official_cards(token, base_url=args.base_url)
        source = f"api:{args.base_url.rstrip('/')}/v1/cards"

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"[saved] payload -> {save_path.resolve()}")

    official_cards = parse_official_cards_payload(payload)
    registry_raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    merged, stats = merge_registry_with_official_cards(registry_raw, official_cards)
    merged["last_sync"] = {"source": source}

    print("[sync-summary]")
    for k, v in stats.items():
        print(f"- {k}: {v}")
    if args.apply:
        registry_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=False), encoding="utf-8")
        print(f"[done] updated registry: {registry_path.resolve()}")
    else:
        print("[dry-run] no files changed. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
