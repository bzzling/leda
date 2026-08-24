#!/usr/bin/env python3
"""Audit and acquire CC-BY PLOS research articles through the official search API."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import heapq
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request


API = "https://api.plos.org/search"
LICENSE_EVIDENCE = "https://journals.plos.org/plosone/s/content-license"
METADATA_FIELDS = "id,doi,journal,article_type,publication_date,subject"
CONTENT_FIELDS = "id,doi,title,abstract,body,journal,article_type,publication_date,subject"


def normalize_doi(value) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    result = str(value).strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        result = result.removeprefix(prefix)
    return result or None


def get_json(parameters: dict[str, str], user_agent: str, attempts: int = 5) -> dict:
    url = API + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except OSError as caught:
            error = caught
            if attempt + 1 != attempts:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def existing_dois(manifests: list[pathlib.Path]) -> set[str]:
    result = set()
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                doi = normalize_doi(json.loads(line).get("doi"))
                if doi:
                    result.add(doi)
    return result


def text_of(document: dict) -> str:
    parts = []
    title = document.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(title.strip())
    for field in ("abstract", "body"):
        value = document.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and item.strip():
                parts.append(" ".join(item.split()))
    return "\n\n".join(parts).strip() + "\n" if parts else ""


def documents(user_agent: str, rows: int):
    cursor = "*"
    query = 'doc_type:full AND article_type:"Research Article"'
    while True:
        result = get_json(
            {
                "q": query,
                "fl": METADATA_FIELDS,
                "rows": str(rows),
                "sort": "id asc",
                "cursorMark": cursor,
                "wt": "json",
            },
            user_agent,
        )
        page = result["response"]["docs"]
        yield from page
        next_cursor = result.get("nextCursorMark")
        if not page or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor


def content_document(identifier: str, user_agent: str) -> dict:
    result = get_json(
        {
            "q": f'id:"{identifier}"',
            "fl": CONTENT_FIELDS,
            "rows": "1",
            "wt": "json",
        },
        user_agent,
    )
    values = result["response"]["docs"]
    if len(values) != 1:
        raise RuntimeError(f"PLOS full document lookup failed: {identifier}")
    return values[0]


def safe_content(identifier: str, user_agent: str) -> tuple[str, dict | None, str | None]:
    try:
        return identifier, content_document(identifier, user_agent), None
    except (OSError, RuntimeError, ValueError) as error:
        return identifier, None, type(error).__name__


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    admitted = []
    rejected = collections.Counter()
    overlap = existing_dois(args.exclude_manifests)
    total_bytes = 0
    scanned = 0
    content_documents_examined = 0
    if args.command == "acquire":
        (output / "documents").mkdir(exist_ok=True)
    selection_path = output / "selection.json"
    selection_config = {
        "seed": args.seed,
        "maximum_documents": args.maximum_documents,
        "exclude_manifest_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.exclude_manifests
        ],
    }
    cached = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if cached and cached.get("config") == selection_config:
        identifiers = cached["identifiers"]
        scanned = int(cached["scanned_documents"])
        rejected.update(cached.get("selection_rejections", {}))
    else:
        selected: list[tuple[int, str]] = []
        seen_dois: set[str] = set()
        for document in documents(args.user_agent, args.page_size):
            scanned += 1
            if scanned % 50_000 == 0:
                print(f"PLOS metadata scanned={scanned}", file=sys.stderr, flush=True)
            doi = normalize_doi(document.get("doi") or document.get("id"))
            if not doi:
                rejected["missing_doi"] += 1
                continue
            if doi in seen_dois:
                rejected["duplicate_api_doi"] += 1
                continue
            seen_dois.add(doi)
            if doi in overlap:
                rejected["identifier_duplicate"] += 1
                continue
            priority = int(hashlib.sha256(f"{args.seed}:{doi}".encode()).hexdigest(), 16)
            item = (-priority, doi)
            if len(selected) < args.maximum_documents:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
        identifiers = [doi for _, doi in sorted(selected, reverse=True)]
        selection_path.write_text(
            json.dumps(
                {
                    "config": selection_config,
                    "scanned_documents": scanned,
                    "identifiers": identifiers,
                    "selection_rejections": dict(sorted(rejected.items())),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    fetch_batch_size = max(16, args.workers * 4)
    target_reached = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(identifiers), fetch_batch_size):
            batch = identifiers[offset : offset + fetch_batch_size]
            fetched = pool.map(lambda value: safe_content(value, args.user_agent), batch)
            for _, document, error in fetched:
                content_documents_examined += 1
                if document is None:
                    rejected["content:" + str(error)] += 1
                    continue
                doi = normalize_doi(document.get("doi") or document.get("id"))
                assert doi is not None
                text = text_of(document)
                encoded = text.encode("utf-8")
                if len(encoded) < args.minimum_bytes:
                    rejected["too_short"] += 1
                    continue
                raw = json.dumps(document, ensure_ascii=False, sort_keys=True).encode()
                record = {
                    "canonical_document_id": "plos:" + doi,
                    "source_family": "PLOS",
                    "source_specific_id": doi,
                    "pmcid": None,
                    "doi": doi,
                    "pmid": None,
                    "article_version": None,
                    "title": document.get("title"),
                    "source_url": "https://doi.org/" + doi,
                    "license": "CC BY",
                    "license_evidence": LICENSE_EVIDENCE,
                    "retrieval_date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                    "raw_metadata_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_metadata_etag": None,
                    "clean_text_bytes": len(encoded),
                    "source_category": document.get("journal"),
                    "source_categories": document.get("subject", []),
                    "article_type": document.get("article_type"),
                }
                if args.command == "acquire":
                    relative = pathlib.Path("documents") / (
                        hashlib.sha256(doi.encode()).hexdigest() + ".txt"
                    )
                    (output / relative).write_bytes(encoded)
                    record["local_document_path"] = relative.as_posix()
                admitted.append(record)
                total_bytes += len(encoded)
                if args.command == "acquire" and args.target_bytes and total_bytes >= args.target_bytes:
                    target_reached = True
                    break
            if target_reached:
                break
    if args.command == "acquire":
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
            for record in admitted:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    journals = collections.Counter(record.get("source_category") or "unknown" for record in admitted)
    categories = collections.Counter(
        category for record in admitted for category in record.get("source_categories", [])
    )
    summary = {
        "mode": args.command,
        "article_query": 'doc_type:full AND article_type:"Research Article"',
        "license": "CC BY",
        "license_evidence": LICENSE_EVIDENCE,
        "scanned_documents": scanned,
        "selected_documents": len(identifiers),
        "content_documents_examined": content_documents_examined,
        "admitted_documents": len(admitted),
        "clean_text_bytes": total_bytes,
        "mean_clean_bytes": total_bytes / len(admitted) if admitted else 0,
        "estimated_tokens_at_4_bytes_per_token": total_bytes // 4,
        "journals": dict(sorted(journals.items())),
        "top_source_categories": dict(categories.most_common(50)),
        "rejections": dict(sorted(rejected.items())),
    }
    (output / f"{args.command}-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "acquire"):
        command = commands.add_parser(name)
        command.add_argument("output", type=pathlib.Path)
        command.add_argument("--exclude-manifests", type=pathlib.Path, nargs="*", default=[])
        command.add_argument("--maximum-documents", type=int, default=1_000 if name == "audit" else 50_000)
        command.add_argument("--target-bytes", type=int, default=0)
        command.add_argument("--seed", type=int, default=3004)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--minimum-bytes", type=int, default=2_000)
        command.add_argument("--page-size", type=int, default=500)
        command.add_argument(
            "--user-agent", default="LedaCorpusResearch/0.1 (https://github.com/bzzling/leda)"
        )
    args = parser.parse_args()
    if args.maximum_documents <= 0 or not 1 <= args.page_size <= 1_000 or not 1 <= args.workers <= 64:
        parser.error("document/page bounds must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
