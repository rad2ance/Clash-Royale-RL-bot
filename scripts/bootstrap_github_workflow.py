from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


DEFAULT_LABELS = [
    ("feature", "0e8a16"),
    ("bug", "d73a4a"),
    ("research", "1d76db"),
    ("infra", "5319e7"),
    ("rl", "0052cc"),
]

DEFAULT_ISSUES = [
    {
        "title": "Simulator: add river/bridge-aware deployment constraints",
        "body": (
            "## Goal\nImprove placement legality by modeling river rows and bridge crossings.\n\n"
            "## Scope\n"
            "- Encode river rows and bridge columns in config\n"
            "- Restrict troop/building deployment accordingly\n"
            "- Keep spells exempt where appropriate\n\n"
            "## Acceptance Criteria\n"
            "- Unit tests cover legal/illegal cases around river and bridges\n"
            "- Existing tests remain green\n"
        ),
        "labels": ["feature", "rl"],
    },
    {
        "title": "Simulator: introduce per-card metadata registry",
        "body": (
            "## Goal\nMap card id to stable metadata (cost/type/targeting) for deterministic behavior.\n\n"
            "## Scope\n"
            "- Add card registry structure\n"
            "- Replace random hand costs with metadata-derived costs\n"
            "- Integrate into legality and dynamics\n\n"
            "## Acceptance Criteria\n"
            "- Card id controls cost/type consistently across episodes\n"
            "- Tests validate metadata-driven behavior\n"
        ),
        "labels": ["feature", "rl"],
    },
    {
        "title": "Training: add checkpoint evaluation script and metrics JSON",
        "body": (
            "## Goal\nMake experiment comparisons reproducible.\n\n"
            "## Scope\n"
            "- Add script to evaluate checkpoints\n"
            "- Save reward, episode length, illegal-action rate, win proxy to JSON\n"
            "- Document usage in README\n"
        ),
        "labels": ["feature", "research"],
    },
    {
        "title": "Data: persist action masks in trajectory files",
        "body": (
            "## Goal\nSupport masked offline analysis and imitation.\n\n"
            "## Scope\n"
            "- Extend episode schema with optional per-step masks\n"
            "- Keep backward compatibility for old files\n"
            "- Add unit tests for load/save paths\n"
        ),
        "labels": ["feature", "rl"],
    },
]


def github_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def create_label(repo: str, token: str, name: str, color: str) -> None:
    url = f"https://api.github.com/repos/{repo}/labels"
    payload = {"name": name, "color": color}
    try:
        github_request("POST", url, token, payload)
        print(f"[label] created: {name}")
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            print(f"[label] exists: {name}")
            return
        raise


def create_issue(repo: str, token: str, title: str, body: str, labels: list[str]) -> str:
    url = f"https://api.github.com/repos/{repo}/issues"
    out = github_request("POST", url, token, {"title": title, "body": body, "labels": labels})
    number = out.get("number")
    html_url = out.get("html_url", "")
    print(f"[issue] #{number} {title}")
    return html_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap GitHub issue workflow for this repo.")
    parser.add_argument("--repo", default="rad2ance/Clash-Royale-RL-bot", help="GitHub repo in owner/name format")
    parser.add_argument("--apply", action="store_true", help="Apply changes remotely (default is dry-run)")
    args = parser.parse_args()

    if not args.apply:
        print("[dry-run] Would create labels:")
        for name, _ in DEFAULT_LABELS:
            print(f"  - {name}")
        print("[dry-run] Would create issues:")
        for item in DEFAULT_ISSUES:
            print(f"  - {item['title']}")
        print("[dry-run] Re-run with --apply and set GITHUB_TOKEN to execute.")
        return

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required when using --apply")

    for name, color in DEFAULT_LABELS:
        create_label(args.repo, token, name, color)

    created_urls: list[str] = []
    for issue in DEFAULT_ISSUES:
        url = create_issue(
            repo=args.repo,
            token=token,
            title=issue["title"],
            body=issue["body"],
            labels=issue["labels"],
        )
        created_urls.append(url)

    print("\n[done] Created issues:")
    for url in created_urls:
        print(f"  - {url}")


if __name__ == "__main__":
    main()
