#!/usr/bin/env python3
"""Collect a broad, identity-free supplement to the frozen E4 Layer-2a pool."""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


OUTDIR = Path(__file__).resolve().parent
EXISTING_POOL = OUTDIR.parent / "candidates_pool.json"
FETCHED_DATE = date.today().isoformat()
USER_AGENT = "igpt-e4-external-supplement/1.0 (research metadata collection)"
GITHUB_REPO = "e2nIEE/pandapower"
WORD_LIMIT = 22


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json, application/json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if isinstance(payload, dict) and payload.get("backoff"):
        time.sleep(float(payload["backoff"]))
    return payload


def html_to_text(value: str | None) -> str:
    parser = _TextExtractor()
    parser.feed(value or "")
    return normalize_space(html.unescape(" ".join(parser.parts)))


def markdown_to_text(value: str | None) -> str:
    text = value or ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+>]\s*", "", text)
    return normalize_space(html.unescape(text))


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def excerpt(value: str, limit: int = WORD_LIMIT) -> str:
    words = normalize_space(value).split()
    return " ".join(words[:limit])


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(value).lower()).strip()


def iso_timestamp(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def fingerprint(record: dict[str, Any]) -> str:
    payload = "\n".join(
        str(record.get(key, "")) for key in ("source_type", "title", "body_excerpt", "url")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def with_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    record["source_fingerprint"] = fingerprint(record)
    return record


def infer_theme(title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    themes = [
        ("timeseries_control", r"time.?series|outputwriter|constcontrol|controller|profile"),
        ("topology_matrix", r"ybus|zbus|admittance|impedance matrix|ptdf|jacobian"),
        ("conversion_io", r"convert|import|export|cim|cgmes|raw file|json|gis"),
        ("unbalanced_short_circuit", r"three.?phase|3ph|unbalanced|short.?circuit|fault"),
        ("opf_optimization", r"\bopf\b|optimal|optim|pyomo|gekko|cost"),
        ("state_estimation", r"state estimation|measurement"),
        ("power_flow", r"power.?flow|runpp|load.?flow|slack"),
        ("plotting", r"plot|colour|color|visual"),
        ("installation_environment", r"install|importerror|qt|numpy|version|environment"),
        ("network_build_modify", r"create_|connect|merge|delete|drop_|network|bus|line"),
    ]
    for theme, pattern in themes:
        if re.search(pattern, text):
            return theme
    return "other"


STACK_TRIAGE: dict[tuple[str, int], tuple[str, str, str | None]] = {
    ("stackoverflow", 78777667): (
        "POSSIBLE_DUPLICATE",
        "cross-source-near-duplicate",
        "github_issue#2352; e4_layer2a:l2a_017",
    ),
    ("stackoverflow", 60838667): (
        "POSSIBLE_DUPLICATE",
        "timeseries-output-retrieval-already-covered",
        "existing E4 OutputWriter/time-series records",
    ),
    ("stackoverflow", 55299203): (
        "POSSIBLE_DUPLICATE",
        "element-deletion-already-covered",
        "existing E4 drop_buses records",
    ),
    ("stackoverflow", 70939178): ("PRIORITY_REVIEW", "direct-domain-api-demand", None),
    ("stackoverflow", 67048257): ("PRIORITY_REVIEW", "direct-matrix-demand", None),
    ("stackoverflow", 71277288): ("PRIORITY_REVIEW", "direct-timeseries-api-demand", None),
    ("stackoverflow", 67344016): ("PRIORITY_REVIEW", "direct-conversion-demand", None),
    ("stackoverflow", 63364203): ("BROAD_REVIEW", "external-data-network-construction", None),
    ("stackoverflow", 70980604): ("LOW_PRIORITY", "image-dependent-missing-context", None),
    ("stackoverflow", 76875889): ("BROAD_REVIEW", "cross-library-optimization", None),
    ("stackoverflow", 67808220): ("BROAD_REVIEW", "plotting-demand", None),
    ("stackoverflow", 72288870): ("BROAD_REVIEW", "plotting-demand", None),
    ("stackoverflow", 60214980): ("LOW_PRIORITY", "generic-control-flow", None),
    ("electronics", 579875): ("PRIORITY_REVIEW", "energy-domain-modeling-question", None),
    ("electronics", 543841): ("PRIORITY_REVIEW", "energy-domain-parameterization-question", None),
}


def default_stack_triage(title: str, body: str) -> tuple[str, str, str | None]:
    text = f"{title} {body}".lower()
    if re.search(r"install|importerror|logging|log file|qt library|sphinx|fmu", text):
        return "LOW_PRIORITY", "environment-or-nonanalysis-task", None
    if re.search(r"how|issue|error|cannot|can't|need", text):
        return "BROAD_REVIEW", "real-user-question-needs-manual-triage", None
    return "LOW_PRIORITY", "weak-direct-pandapower-demand", None


def collect_stackexchange() -> list[dict[str, Any]]:
    endpoint = "https://api.stackexchange.com/2.3"
    searches = [
        (
            "stackoverflow",
            "stackoverflow_question",
            "tag:pandapower",
            "/questions",
            {"site": "stackoverflow", "tagged": "pandapower"},
        ),
        (
            "stackoverflow",
            "stackoverflow_question",
            "fulltext:pandapower",
            "/search/advanced",
            {"site": "stackoverflow", "q": "pandapower"},
        ),
        (
            "electronics",
            "stackexchange_electronics_question",
            "fulltext:pandapower",
            "/search/advanced",
            {"site": "electronics", "q": "pandapower"},
        ),
    ]
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for site, source_type, found_by, path, search_params in searches:
        params = {
            **search_params,
            "pagesize": 100,
            "page": 1,
            "order": "desc",
            "sort": "creation",
            "filter": "withbody",
        }
        while True:
            payload = request_json(f"{endpoint}{path}", params)
            for item in payload["items"]:
                key = (site, int(item["question_id"]))
                if key not in merged:
                    title = html.unescape(item["title"])
                    body = html_to_text(item.get("body"))
                    decision, reason, duplicate_of = STACK_TRIAGE.get(
                        key, default_stack_triage(title, body)
                    )
                    record = {
                        "candidate_id": f"{site}:{item['question_id']}",
                        "source_type": source_type,
                        "question_id": int(item["question_id"]),
                        "title": title,
                        "body_excerpt": excerpt(body),
                        "url": item["link"],
                        "published_at": iso_timestamp(item.get("creation_date")),
                        "fetched_date": FETCHED_DATE,
                        "tags": item.get("tags", []),
                        "score": item.get("score"),
                        "answer_count": item.get("answer_count"),
                        "is_answered": item.get("is_answered"),
                        "found_by_paths": [],
                        "theme": infer_theme(title, body),
                        "screening_decision": decision,
                        "reason_code": reason,
                        "duplicate_of": duplicate_of,
                        "provenance_note": (
                            "Stack Exchange CC BY-SA terms vary by post date; "
                            "source URL retained for attribution."
                        ),
                    }
                    merged[key] = record
                merged[key]["found_by_paths"].append(found_by)
            if not payload.get("has_more"):
                break
            params["page"] += 1
    return [
        with_fingerprint(record)
        for record in sorted(merged.values(), key=lambda row: row["candidate_id"])
    ]


def openmod_triage(topic_id: int, title: str, body: str) -> tuple[str, str]:
    priority_ids = {2706, 4026, 4979}
    broad_ids = {1422, 1473, 1642, 4741, 5033}
    if topic_id in priority_ids:
        return "PRIORITY_REVIEW", "direct-engineering-workflow-or-analysis-question"
    if topic_id in broad_ids:
        return "BROAD_REVIEW", "cross-tool-or-model-construction-discussion"
    if re.search(r"\?|issue|problem|how|import|integrat", f"{title} {body}", re.I):
        return "BROAD_REVIEW", "real-user-discussion-needs-manual-triage"
    return "LOW_PRIORITY", "coordination-or-nondemand-discussion"


def collect_openmod() -> list[dict[str, Any]]:
    base = "https://forum.openmod.org"
    tag_payload = request_json(f"{base}/tag/pandapower.json")
    search_payload = request_json(f"{base}/search.json", {"q": "pandapower"})
    tag_ids = {int(topic["id"]) for topic in tag_payload["topic_list"]["topics"]}
    topics = {
        int(topic["id"]): topic
        for topic in [
            *tag_payload["topic_list"]["topics"],
            *search_payload.get("topics", []),
        ]
    }
    records: list[dict[str, Any]] = []
    for topic_id, topic in sorted(topics.items()):
        detail = request_json(f"{base}/t/{topic['slug']}/{topic_id}.json")
        posts = detail.get("post_stream", {}).get("posts", [])
        body = html_to_text(posts[0].get("cooked")) if posts else ""
        decision, reason = openmod_triage(topic_id, topic["title"], body)
        found_by = ["fulltext:pandapower"]
        if topic_id in tag_ids:
            found_by.insert(0, "tag:pandapower")
        records.append(
            with_fingerprint(
                {
                    "candidate_id": f"openmod:{topic_id}",
                    "source_type": "openmod_topic",
                    "topic_id": topic_id,
                    "title": topic["title"],
                    "body_excerpt": excerpt(body),
                    "url": f"{base}/t/{topic['slug']}/{topic_id}",
                    "published_at": topic.get("created_at"),
                    "fetched_date": FETCHED_DATE,
                    "tags": topic.get("tags", []),
                    "posts_count": topic.get("posts_count"),
                    "found_by_paths": found_by,
                    "theme": infer_theme(topic["title"], body),
                    "screening_decision": decision,
                    "reason_code": reason,
                    "duplicate_of": None,
                    "provenance_note": (
                        "Public Discourse topic; only title, short excerpt, and URL retained. "
                        "Check forum terms before redistributing source text."
                    ),
                }
            )
        )
        time.sleep(0.1)
    return records


def load_existing_pool() -> tuple[set[str], set[int], dict[str, str]]:
    records = json.loads(EXISTING_POOL.read_text(encoding="utf-8"))
    urls = {record["url"] for record in records}
    discussion_numbers = {
        int(record["issue_or_discussion_number"])
        for record in records
        if record["source_type"] == "github_discussion"
    }
    titles = {normalize_title(record["title"]): record["url"] for record in records}
    return urls, discussion_numbers, titles


def github_paginated(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = request_json(
            f"https://api.github.com{path}", {**params, "per_page": 100, "page": page}
        )
        batch = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        rows.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if path == "/search/issues":
            time.sleep(6.2)
    return rows


def collect_github_discussions() -> list[dict[str, Any]]:
    existing_urls, existing_numbers, existing_titles = load_existing_pool()
    rows = github_paginated(f"/repos/{GITHUB_REPO}/discussions", {})
    records: list[dict[str, Any]] = []
    for item in rows:
        number = int(item["number"])
        title = item["title"]
        body = markdown_to_text(item.get("body"))
        normalized = normalize_title(title)
        exact_existing_match = item["html_url"] in existing_urls or number in existing_numbers
        title_match = existing_titles.get(normalized)
        category = (item.get("category") or {}).get("name", "")
        if exact_existing_match:
            decision, reason = "ALREADY_PRESENT", "already-in-frozen-candidate-pool"
            duplicate_of = item["html_url"]
        elif title_match:
            decision, reason = "POSSIBLE_DUPLICATE", "normalized-title-match"
            duplicate_of = title_match
        elif not body and len(title.strip()) < 5:
            decision, reason = "LOW_PRIORITY", "empty-or-nonsubstantive-post"
            duplicate_of = None
        elif category == "Q&A" or re.search(r"\?|how|issue|problem|error|fail|wrong", title, re.I):
            decision, reason = "PRIORITY_REVIEW", "previously-uncollected-user-discussion"
            duplicate_of = None
        else:
            decision, reason = "BROAD_REVIEW", "previously-uncollected-project-discussion"
            duplicate_of = None
        records.append(
            with_fingerprint(
                {
                    "candidate_id": f"github_discussion:{number}",
                    "source_type": "github_discussion",
                    "repository": GITHUB_REPO,
                    "issue_or_discussion_number": number,
                    "title": title,
                    "body_excerpt": excerpt(body),
                    "url": item["html_url"],
                    "published_at": item.get("created_at"),
                    "fetched_date": FETCHED_DATE,
                    "category": category,
                    "comments_count": item.get("comments"),
                    "has_chosen_answer": bool(item.get("answer_chosen_at")),
                    "found_by_paths": ["repository:all_discussions"],
                    "theme": infer_theme(title, body),
                    "screening_decision": decision,
                    "reason_code": reason,
                    "duplicate_of": duplicate_of,
                    "provenance_note": (
                        "Public GitHub discussion; only metadata, short excerpt, and URL retained."
                    ),
                }
            )
        )
    return sorted(records, key=lambda row: row["issue_or_discussion_number"])


def crossrepo_triage(title: str, body: str) -> tuple[str, str]:
    text = f"{title} {body}".lower()
    if re.search(r"dependabot|bump |release|changelog|ci failure|test matrix|roadmap|meta:", text):
        return "LOW_PRIORITY", "maintenance-or-release-signal"
    if re.search(r"\?|how|cannot|can't|error|wrong|fail|problem|support|convert|import", text):
        return "PRIORITY_REVIEW", "cross-ecosystem-user-or-interoperability-demand"
    return "BROAD_REVIEW", "cross-ecosystem-pandapower-mention"


def collect_github_crossrepo() -> list[dict[str, Any]]:
    existing_urls, _, existing_titles = load_existing_pool()
    query = f"pandapower in:title,body is:issue -repo:{GITHUB_REPO}"
    rows = github_paginated(
        "/search/issues",
        {"q": query, "sort": "created", "order": "desc"},
    )
    records: list[dict[str, Any]] = []
    for item in rows:
        if "pull_request" in item:
            continue
        title = item["title"]
        body = markdown_to_text(item.get("body"))
        repository = item["repository_url"].split("/repos/", 1)[-1]
        normalized = normalize_title(title)
        duplicate_of = None
        if item["html_url"] in existing_urls:
            decision, reason = "ALREADY_PRESENT", "already-in-frozen-candidate-pool"
            duplicate_of = item["html_url"]
        elif normalized in existing_titles:
            decision, reason = "POSSIBLE_DUPLICATE", "normalized-title-match"
            duplicate_of = existing_titles[normalized]
        else:
            decision, reason = crossrepo_triage(title, body)
        records.append(
            with_fingerprint(
                {
                    "candidate_id": f"github_crossrepo:{repository}#{item['number']}",
                    "source_type": "github_crossrepo_issue",
                    "repository": repository,
                    "issue_or_discussion_number": int(item["number"]),
                    "title": title,
                    "body_excerpt": excerpt(body),
                    "url": item["html_url"],
                    "published_at": item.get("created_at"),
                    "fetched_date": FETCHED_DATE,
                    "state": item.get("state"),
                    "comments_count": item.get("comments"),
                    "found_by_paths": ["github_search:pandapower-in-title-or-body"],
                    "theme": infer_theme(title, body),
                    "screening_decision": decision,
                    "reason_code": reason,
                    "duplicate_of": duplicate_of,
                    "provenance_note": (
                        "Public GitHub issue outside the pandapower repository; only metadata, "
                        "short excerpt, and URL retained."
                    ),
                }
            )
        )
    return sorted(records, key=lambda row: row["candidate_id"].lower())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "by_source_type": dict(sorted(Counter(row["source_type"] for row in records).items())),
        "by_screening_decision": dict(
            sorted(Counter(row["screening_decision"] for row in records).items())
        ),
        "by_theme": dict(sorted(Counter(row["theme"] for row in records).items())),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stackexchange = collect_stackexchange()
    openmod = collect_openmod()
    discussions = collect_github_discussions()
    crossrepo = collect_github_crossrepo()

    outputs = {
        "stackexchange_candidates.json": stackexchange,
        "openmod_candidates.json": openmod,
        "github_discussions_inventory.json": discussions,
        "github_crossrepo_candidates.json": crossrepo,
    }
    for filename, payload in outputs.items():
        write_json(OUTDIR / filename, payload)

    supplement = [
        *stackexchange,
        *openmod,
        *(row for row in discussions if row["screening_decision"] != "ALREADY_PRESENT"),
        *(row for row in crossrepo if row["screening_decision"] != "ALREADY_PRESENT"),
    ]
    supplement.sort(key=lambda row: row["candidate_id"].lower())
    write_json(OUTDIR / "all_external_candidates.json", supplement)

    community_types = {
        "stackoverflow_question",
        "stackexchange_electronics_question",
        "openmod_topic",
        "github_discussion",
    }
    community_shortlist = [
        row
        for row in supplement
        if row["source_type"] in community_types
        and row["screening_decision"] == "PRIORITY_REVIEW"
    ]
    community_shortlist.sort(
        key=lambda row: (
            row["source_type"],
            row.get("published_at") or "",
            row["candidate_id"],
        )
    )
    write_json(OUTDIR / "community_priority_shortlist.json", community_shortlist)

    manifest = {
        "schema_version": "1.0",
        "fetched_date": FETCHED_DATE,
        "purpose": (
            "Broad exploratory supplement for E4 external-demand collection. "
            "It does not modify the frozen 84-record Layer-2a evaluation set."
        ),
        "privacy": (
            "No usernames, avatars, email addresses, comments, answers, or full post bodies "
            "are stored. Body excerpts are capped at 22 whitespace-delimited words."
        ),
        "sources": {
            "stack_exchange": {
                "endpoints": [
                    "https://api.stackexchange.com/2.3/questions?site=stackoverflow&tagged=pandapower",
                    "https://api.stackexchange.com/2.3/search/advanced?site=stackoverflow&q=pandapower",
                    "https://api.stackexchange.com/2.3/search/advanced?site=electronics&q=pandapower",
                ],
                "summary": summarize(stackexchange),
            },
            "openmod": {
                "endpoints": [
                    "https://forum.openmod.org/tag/pandapower.json",
                    "https://forum.openmod.org/search.json?q=pandapower",
                ],
                "summary": summarize(openmod),
            },
            "github_discussions": {
                "endpoint": f"https://api.github.com/repos/{GITHUB_REPO}/discussions",
                "summary": summarize(discussions),
                "note": "Inventory includes existing-pool matches; merged supplement omits them.",
            },
            "github_crossrepo": {
                "endpoint": "https://api.github.com/search/issues",
                "query": f"pandapower in:title,body is:issue -repo:{GITHUB_REPO}",
                "summary": summarize(crossrepo),
            },
        },
        "merged_supplement": summarize(supplement),
        "community_priority_shortlist": summarize(community_shortlist),
        "outputs": {},
    }
    for filename in [
        *outputs,
        "all_external_candidates.json",
        "community_priority_shortlist.json",
    ]:
        path = OUTDIR / filename
        manifest["outputs"][filename] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    write_json(OUTDIR / "manifest.json", manifest)
    print(json.dumps(manifest["merged_supplement"], indent=2))


if __name__ == "__main__":
    main()
