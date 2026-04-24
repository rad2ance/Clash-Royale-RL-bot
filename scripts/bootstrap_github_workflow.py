from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


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


def github_request(method: str, url: str, token: str, payload: dict | None = None) -> Any:
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


def github_graphql_request(token: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"query": query, "variables": variables or {}}
    out = github_request("POST", "https://api.github.com/graphql", token, payload)
    if "errors" in out and out["errors"]:
        raise RuntimeError(f"GitHub GraphQL error: {out['errors']}")
    data = out.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("GitHub GraphQL response missing 'data'")
    return data


def normalize_issue_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def list_existing_issues(repo: str, token: str, state: str = "all") -> dict[str, dict[str, Any]]:
    issues: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page=100&page={page}"
        rows = github_request("GET", url, token)
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            # GitHub issues API includes PRs; skip those.
            if isinstance(row, dict) and "pull_request" in row:
                continue
            title = str(row.get("title", "")).strip() if isinstance(row, dict) else ""
            if title:
                normalized = normalize_issue_title(title)
                issues[normalized] = {
                    "number": row.get("number"),
                    "title": title,
                    "html_url": row.get("html_url", ""),
                    "node_id": row.get("node_id", ""),
                }
        page += 1
    return issues


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


def create_issue(repo: str, token: str, title: str, body: str, labels: list[str]) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/issues"
    out = github_request("POST", url, token, {"title": title, "body": body, "labels": labels})
    number = out.get("number")
    html_url = out.get("html_url", "")
    print(f"[issue] #{number} {title}")
    return {
        "number": number,
        "title": title,
        "html_url": html_url,
        "node_id": out.get("node_id", ""),
    }


def resolve_project_owner(owner_login: str | None, token: str) -> tuple[str, str]:
    if owner_login:
        query = """
        query($login: String!) {
          user(login: $login) { id login }
          organization(login: $login) { id login }
        }
        """
        data = github_graphql_request(token, query, {"login": owner_login})
        user = data.get("user")
        if isinstance(user, dict) and user.get("id"):
            return str(user["id"]), str(user.get("login", owner_login))
        org = data.get("organization")
        if isinstance(org, dict) and org.get("id"):
            return str(org["id"]), str(org.get("login", owner_login))
        raise RuntimeError(f"Could not resolve GitHub owner '{owner_login}'")

    query = """
    query {
      viewer { login id }
    }
    """
    data = github_graphql_request(token, query)
    viewer = data.get("viewer")
    if not isinstance(viewer, dict) or not viewer.get("id") or not viewer.get("login"):
        raise RuntimeError("Could not resolve GraphQL viewer for default project owner")
    return str(viewer["id"]), str(viewer["login"])


def find_or_create_project(owner_id: str, title: str, token: str) -> dict[str, Any]:
    query = """
    query($ownerId: ID!, $cursor: String) {
      node(id: $ownerId) {
        ... on User {
          projectsV2(first: 50, after: $cursor) {
            nodes { id title url number }
            pageInfo { hasNextPage endCursor }
          }
        }
        ... on Organization {
          projectsV2(first: 50, after: $cursor) {
            nodes { id title url number }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }
    """
    cursor: str | None = None
    while True:
        data = github_graphql_request(token, query, {"ownerId": owner_id, "cursor": cursor})
        node = data.get("node", {})
        projects = {}
        if isinstance(node, dict):
            projects = node.get("projectsV2", {})
        if not isinstance(projects, dict):
            break
        for project in projects.get("nodes", []) or []:
            if isinstance(project, dict) and project.get("title") == title:
                return project
        page_info = projects.get("pageInfo", {})
        if not isinstance(page_info, dict) or not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")

    mutation = """
    mutation($ownerId: ID!, $title: String!) {
      createProjectV2(input: {ownerId: $ownerId, title: $title}) {
        projectV2 { id title url number }
      }
    }
    """
    data = github_graphql_request(token, mutation, {"ownerId": owner_id, "title": title})
    created = data.get("createProjectV2", {}).get("projectV2", {})
    if not isinstance(created, dict) or not created.get("id"):
        raise RuntimeError("Failed to create ProjectV2")
    print(f"[project] created: {created.get('title')} ({created.get('url', 'n/a')})")
    return created


def list_project_items_and_fields(project_id: str, token: str) -> tuple[set[str], str | None, dict[str, str]]:
    query = """
    query($projectId: ID!, $cursor: String) {
      node(id: $projectId) {
        ... on ProjectV2 {
          items(first: 100, after: $cursor) {
            nodes {
              id
              content {
                ... on Issue { id }
                ... on PullRequest { id }
              }
            }
            pageInfo { hasNextPage endCursor }
          }
          fields(first: 50) {
            nodes {
              ... on ProjectV2SingleSelectField {
                id
                name
                options { id name }
              }
            }
          }
        }
      }
    }
    """
    content_ids: set[str] = set()
    status_field_id: str | None = None
    status_options: dict[str, str] = {}
    cursor: str | None = None
    while True:
        data = github_graphql_request(token, query, {"projectId": project_id, "cursor": cursor})
        node = data.get("node", {})
        if not isinstance(node, dict):
            break
        items = node.get("items", {})
        if isinstance(items, dict):
            for item in items.get("nodes", []) or []:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", {})
                if isinstance(content, dict):
                    content_id = content.get("id")
                    if isinstance(content_id, str) and content_id:
                        content_ids.add(content_id)
            page_info = items.get("pageInfo", {})
            if isinstance(page_info, dict) and page_info.get("hasNextPage"):
                cursor = page_info.get("endCursor")
            else:
                cursor = None

        fields = node.get("fields", {})
        if isinstance(fields, dict):
            for field in fields.get("nodes", []) or []:
                if not isinstance(field, dict):
                    continue
                name = str(field.get("name", "")).strip().lower()
                if name == "status":
                    status_field_id = field.get("id")
                    status_options = {
                        str(opt.get("name", "")).strip().lower(): str(opt.get("id", ""))
                        for opt in (field.get("options", []) or [])
                        if isinstance(opt, dict) and opt.get("name") and opt.get("id")
                    }
                    break
        if status_field_id or cursor is None:
            if cursor is None:
                break
        if cursor is None:
            break
    return content_ids, status_field_id, status_options


def add_issue_to_project(project_id: str, issue_node_id: str, token: str) -> str:
    mutation = """
    mutation($projectId: ID!, $contentId: ID!) {
      addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
        item { id }
      }
    }
    """
    data = github_graphql_request(token, mutation, {"projectId": project_id, "contentId": issue_node_id})
    item_id = data.get("addProjectV2ItemById", {}).get("item", {}).get("id", "")
    if not item_id:
        raise RuntimeError("Failed to add issue to project")
    return str(item_id)


def set_project_item_status(
    project_id: str,
    item_id: str,
    status_field_id: str,
    status_option_id: str,
    token: str,
) -> None:
    mutation = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(
        input: {
          projectId: $projectId
          itemId: $itemId
          fieldId: $fieldId
          value: {singleSelectOptionId: $optionId}
        }
      ) {
        projectV2Item { id }
      }
    }
    """
    github_graphql_request(
        token,
        mutation,
        {
            "projectId": project_id,
            "itemId": item_id,
            "fieldId": status_field_id,
            "optionId": status_option_id,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap GitHub issue workflow for this repo.")
    parser.add_argument("--repo", default="rad2ance/Clash-Royale-RL-bot", help="GitHub repo in owner/name format")
    parser.add_argument("--apply", action="store_true", help="Apply changes remotely (default is dry-run)")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Create issues even if the same title already exists.",
    )
    parser.add_argument(
        "--sync-project",
        action="store_true",
        help="Sync managed issues into a GitHub ProjectV2 board.",
    )
    parser.add_argument(
        "--project-title",
        default="Clash Royale RL Bot",
        help="ProjectV2 title to find or create when --sync-project is set.",
    )
    parser.add_argument(
        "--project-owner",
        default="",
        help="GitHub login owning the project (user/org). Defaults to authenticated viewer.",
    )
    parser.add_argument(
        "--project-status",
        default="Todo",
        help="Single-select status to set on newly added project items (best-effort).",
    )
    args = parser.parse_args()

    if not args.apply:
        print("[dry-run] Would create labels:")
        for name, _ in DEFAULT_LABELS:
            print(f"  - {name}")
        print("[dry-run] Would create issues:")
        for item in DEFAULT_ISSUES:
            print(f"  - {item['title']}")
        if args.sync_project:
            owner_note = args.project_owner or "<viewer>"
            print(
                f"[dry-run] Would sync issues to project '{args.project_title}' "
                f"(owner: {owner_note}, status: {args.project_status})."
            )
        print("[dry-run] Re-run with --apply and set GITHUB_TOKEN to execute.")
        return

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required when using --apply")

    for name, color in DEFAULT_LABELS:
        create_label(args.repo, token, name, color)

    existing_issues: dict[str, dict[str, Any]] = {}
    if not args.allow_duplicates:
        existing_issues = list_existing_issues(args.repo, token, state="all")

    created_urls: list[str] = []
    skipped_titles: list[str] = []
    managed_issues: list[dict[str, Any]] = []
    for issue in DEFAULT_ISSUES:
        normalized = normalize_issue_title(issue["title"])
        if not args.allow_duplicates and normalized in existing_issues:
            print(f"[issue] exists, skipping: {issue['title']}")
            skipped_titles.append(issue["title"])
            managed_issues.append(existing_issues[normalized])
            continue
        created = create_issue(
            repo=args.repo,
            token=token,
            title=issue["title"],
            body=issue["body"],
            labels=issue["labels"],
        )
        created_urls.append(created["html_url"])
        existing_issues[normalized] = created
        managed_issues.append(created)

    if args.sync_project:
        owner_id, owner_login = resolve_project_owner(args.project_owner or None, token)
        project = find_or_create_project(owner_id, args.project_title, token)
        project_id = str(project["id"])
        project_url = str(project.get("url", ""))
        print(f"[project] using: {args.project_title} ({project_url or 'url unavailable'})")

        existing_content_ids, status_field_id, status_options = list_project_items_and_fields(project_id, token)
        target_status_key = args.project_status.strip().lower()
        target_status_option_id = status_options.get(target_status_key) if status_field_id else None

        added_count = 0
        already_present_count = 0
        status_set_count = 0
        for issue in managed_issues:
            node_id = str(issue.get("node_id", "")).strip()
            if not node_id:
                print(f"[project] skipping issue with missing node_id: {issue.get('title', '<unknown>')}")
                continue
            if node_id in existing_content_ids:
                already_present_count += 1
                continue
            item_id = add_issue_to_project(project_id, node_id, token)
            added_count += 1
            existing_content_ids.add(node_id)
            if status_field_id and target_status_option_id:
                set_project_item_status(
                    project_id=project_id,
                    item_id=item_id,
                    status_field_id=status_field_id,
                    status_option_id=target_status_option_id,
                    token=token,
                )
                status_set_count += 1

        if args.project_status and status_field_id and not target_status_option_id:
            print(
                f"[project] status option '{args.project_status}' not found "
                "in project Status field; skipped status assignment."
            )
        if args.project_status and not status_field_id:
            print("[project] Status field not found; skipped status assignment.")

        print(
            f"[project] sync complete for owner '{owner_login}': "
            f"added={added_count}, already_present={already_present_count}, status_set={status_set_count}"
        )

    print("\n[done] Created issues:")
    if not created_urls:
        print("  - none")
    else:
        for url in created_urls:
            print(f"  - {url}")
    if skipped_titles:
        print("[done] Skipped existing titles:")
        for title in skipped_titles:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
