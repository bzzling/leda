#!/usr/bin/env python3
"""Build a provenance-tracked CC0/CC-BY PMC text corpus through OAI-PMH."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


OAI_ENDPOINT = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
EXCLUDED_ELEMENTS = {
    "ack",
    "app-group",
    "fn-group",
    "notes",
    "permissions",
    "ref-list",
    "supplementary-material",
}
TEXT_ELEMENTS = {"article-title", "caption", "p", "title"}
SPACE = re.compile(r"\s+")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_fragment(element: ET.Element) -> str:
    return SPACE.sub(" ", "".join(element.itertext())).strip()


def license_class(article: ET.Element) -> tuple[str | None, str]:
    candidates: list[str] = []
    for element in article.iter():
        if local_name(element.tag) != "license":
            continue
        candidates.extend(str(value) for value in element.attrib.values())
        text = clean_fragment(element)
        if text:
            candidates.append(text)
    evidence = " | ".join(dict.fromkeys(candidates))
    folded = evidence.casefold()
    if "creativecommons.org/publicdomain/zero/" in folded or re.search(r"\bcc0\b", folded):
        return "CC0", evidence
    by_url = "creativecommons.org/licenses/by/" in folded
    by_text = bool(re.search(r"\bcc[ -]?by\b", folded))
    restricted = any(term in folded for term in ("by-nc", "by-nd", "by-sa", "noncommercial"))
    if (by_url or by_text) and not restricted:
        return "CC BY", evidence
    return None, evidence or "missing"


def article_id(article: ET.Element, kind: str) -> str | None:
    for element in article.iter():
        if local_name(element.tag) == "article-id" and element.attrib.get("pub-id-type") == kind:
            value = clean_fragment(element)
            if value:
                return value
    return None


def subjects(article: ET.Element) -> list[str]:
    values = []
    for element in article.iter():
        if local_name(element.tag) == "subject":
            value = clean_fragment(element)
            if value and value not in values:
                values.append(value)
    return values


def extract_text(article: ET.Element) -> str:
    paragraphs: list[str] = []

    def visit(element: ET.Element, excluded: bool = False) -> None:
        name = local_name(element.tag)
        excluded = excluded or name in EXCLUDED_ELEMENTS
        if excluded:
            return
        if name in TEXT_ELEMENTS:
            text = clean_fragment(element)
            if text and (not paragraphs or paragraphs[-1] != text):
                paragraphs.append(text)
            return
        for child in element:
            visit(child, excluded)

    front = next((item for item in article if local_name(item.tag) == "front"), None)
    body = next((item for item in article if local_name(item.tag) == "body"), None)
    if front is not None:
        for element in front.iter():
            if local_name(element.tag) in {"article-title", "abstract"}:
                visit(element)
    if body is not None:
        visit(body)
    return "\n\n".join(paragraphs).strip() + "\n" if paragraphs else ""


def request_xml(parameters: dict[str, str], user_agent: str, attempts: int = 5) -> ET.Element:
    url = OAI_ENDPOINT + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return ET.fromstring(response.read())
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def existing_records(manifest: pathlib.Path) -> tuple[set[str], set[str], int, int]:
    source_ids: set[str] = set()
    checksums: set[str] = set()
    byte_count = 0
    document_count = 0
    if not manifest.exists():
        return source_ids, checksums, byte_count, document_count
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            source_ids.add(record["stable_source_id"])
            checksums.add(record["content_checksum"])
            byte_count += int(record["content_bytes"])
            document_count += 1
    return source_ids, checksums, byte_count, document_count


def write_path_manifests(output: pathlib.Path, provenance: pathlib.Path) -> collections.Counter:
    split_paths: dict[str, list[str]] = {"train": [], "validation": []}
    licenses: collections.Counter = collections.Counter()
    with provenance.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            split_paths[record["split"]].append(record["document_path"])
            licenses[record["license"]] += 1
    for split, paths in split_paths.items():
        paths.sort()
        (output / f"{split}.paths").write_text("\n".join(paths) + "\n", encoding="utf-8")
    return licenses


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    document_root = output / "documents"
    (document_root / "train").mkdir(parents=True, exist_ok=True)
    (document_root / "validation").mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance.jsonl"
    known_ids, known_checksums, total_bytes, total_documents = existing_records(provenance)
    skipped: collections.Counter = collections.Counter()
    duplicates = 0
    retrieved = 0
    retrieval_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    current_date = datetime.date.fromisoformat(args.from_date)
    final_date = datetime.date.fromisoformat(args.until_date)
    if current_date > final_date:
        raise ValueError("from-date must not follow until-date")
    with provenance.open("a", encoding="utf-8") as manifest:
        stop = False
        while current_date <= final_date and not stop:
            day = current_date.isoformat()
            parameters = {
                "verb": "ListRecords",
                "metadataPrefix": "pmc",
                "set": "pmc-open",
                "from": day,
                "until": day,
            }
            while not stop:
                root = request_xml(parameters, args.user_agent)
                error = next((item for item in root.iter() if local_name(item.tag) == "error"), None)
                if error is not None:
                    if error.attrib.get("code") == "noRecordsMatch":
                        break
                    raise RuntimeError(clean_fragment(error))
                records = [item for item in root.iter() if local_name(item.tag) == "record"]
                if not records:
                    break
                for record in records:
                    article = next(
                        (item for item in record.iter() if local_name(item.tag) == "article"), None
                    )
                    if article is None:
                        skipped["missing_article"] += 1
                        continue
                    pmcid = article_id(article, "pmcid")
                    if not pmcid:
                        skipped["missing_pmcid"] += 1
                        continue
                    pmcid = pmcid if pmcid.startswith("PMC") else "PMC" + pmcid
                    if pmcid in known_ids:
                        continue
                    retrieved += 1
                    license_name, license_evidence = license_class(article)
                    if license_name is None:
                        skipped["license:" + license_evidence[:160]] += 1
                        continue
                    content = extract_text(article)
                    if len(content.encode("utf-8")) < args.minimum_bytes:
                        skipped["too_short"] += 1
                        continue
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    checksum = "sha256:" + digest
                    if checksum in known_checksums:
                        duplicates += 1
                        known_ids.add(pmcid)
                        continue
                    split_value = int(digest[:16], 16) % 10_000
                    split = "validation" if split_value < args.validation_permyriad else "train"
                    relative_path = pathlib.Path("documents") / split / f"{pmcid}.txt"
                    final_path = output / relative_path
                    temporary_path = final_path.with_suffix(".txt.tmp")
                    temporary_path.write_text(content, encoding="utf-8")
                    temporary_path.replace(final_path)
                    content_bytes = len(content.encode("utf-8"))
                    metadata = {
                        "source_family": "PMC OAI-PMH",
                        "stable_source_id": pmcid,
                        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                        "official_identifier": pmcid,
                        "doi": article_id(article, "doi"),
                        "license": license_name,
                        "license_evidence": license_evidence,
                        "retrieval_date": retrieval_date,
                        "content_checksum": checksum,
                        "content_bytes": content_bytes,
                        "science_domain_category": subjects(article),
                        "split": split,
                        "document_path": str(final_path),
                    }
                    manifest.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                    manifest.flush()
                    known_ids.add(pmcid)
                    known_checksums.add(checksum)
                    total_bytes += content_bytes
                    total_documents += 1
                    if total_documents % 100 == 0:
                        print(
                            f"admitted={total_documents} bytes={total_bytes} retrieved={retrieved}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if total_bytes >= args.target_bytes or (
                        args.max_documents and total_documents >= args.max_documents
                    ):
                        stop = True
                        break
                if stop:
                    break
                token_element = next(
                    (item for item in root.iter() if local_name(item.tag) == "resumptionToken"), None
                )
                token = clean_fragment(token_element) if token_element is not None else ""
                if not token:
                    break
                parameters = {"verb": "ListRecords", "resumptionToken": token}
                time.sleep(args.delay_seconds)
            current_date += datetime.timedelta(days=1)
    license_counts = write_path_manifests(output, provenance)
    summary = {
        "admitted_documents": total_documents,
        "cleaned_bytes": total_bytes,
        "exact_duplicates_removed_this_run": duplicates,
        "licenses": dict(sorted(license_counts.items())),
        "retrieved_records_this_run": retrieved,
        "skipped_this_run": dict(sorted(skipped.items())),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest allowlisted PMC full text through the official OAI-PMH API."
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--from-date", required=True, help="OAI lower date bound, YYYY-MM-DD")
    parser.add_argument("--until-date", required=True, help="OAI upper date bound, YYYY-MM-DD")
    parser.add_argument("--target-bytes", type=int, default=140_000_000)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--minimum-bytes", type=int, default=2_000)
    parser.add_argument("--validation-permyriad", type=int, default=100)
    parser.add_argument("--delay-seconds", type=float, default=0.4)
    parser.add_argument(
        "--user-agent",
        default="LedaCorpusResearch/0.1 (https://github.com/bzzling/leda)",
    )
    args = parser.parse_args()
    if args.target_bytes <= 0 or args.max_documents < 0 or args.minimum_bytes < 0:
        parser.error("byte/document limits must be nonnegative and target bytes must be positive")
    if not 0 < args.validation_permyriad < 10_000:
        parser.error("validation-permyriad must be in [1, 9999]")
    run(args)


if __name__ == "__main__":
    main()
