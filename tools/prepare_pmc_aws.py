#!/usr/bin/env python3
"""Build a provenance-tracked CC0/CC-BY corpus from PMC's official AWS dataset."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from prepare_pmc import extract_text, local_name, subjects, write_path_manifests


BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com/"
ALLOWLIST = {"CC0", "CC BY"}


def get_bytes(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def list_prefixes(prefix: str, continuation: str | None, user_agent: str) -> tuple[list[str], str | None]:
    parameters = {"list-type": "2", "delimiter": "/", "prefix": prefix, "max-keys": "1000"}
    if continuation:
        parameters["continuation-token"] = continuation
    root = ET.fromstring(get_bytes(BUCKET_URL + "?" + urllib.parse.urlencode(parameters), user_agent))
    prefixes = [
        "".join(item.itertext()).strip()
        for item in root.iter()
        if local_name(item.tag) == "Prefix" and "".join(item.itertext()).strip().startswith("PMC")
    ]
    next_token = next(
        (
            "".join(item.itertext()).strip()
            for item in root.iter()
            if local_name(item.tag) == "NextContinuationToken"
        ),
        None,
    )
    return prefixes, next_token


def fetch_metadata(prefix: str, user_agent: str) -> tuple[str, dict | None, str | None]:
    stem = prefix.rstrip("/")
    url = BUCKET_URL + "metadata/" + urllib.parse.quote(stem + ".json")
    try:
        return prefix, json.loads(get_bytes(url, user_agent)), None
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        return prefix, None, type(error).__name__


def fetch_article(item: tuple[str, dict], user_agent: str) -> tuple[str, dict, ET.Element | None, str | None]:
    prefix, metadata = item
    xml_url = metadata.get("xml_url")
    if not isinstance(xml_url, str) or not xml_url.startswith("s3://pmc-oa-opendata/"):
        return prefix, metadata, None, "missing_xml"
    key = xml_url.split("?", 1)[0].removeprefix("s3://pmc-oa-opendata/")
    try:
        return prefix, metadata, ET.fromstring(get_bytes(BUCKET_URL + urllib.parse.quote(key), user_agent)), None
    except (OSError, ET.ParseError, urllib.error.HTTPError) as error:
        return prefix, metadata, None, type(error).__name__


def existing_records(manifest: pathlib.Path) -> tuple[set[str], set[str], int, int]:
    ids: set[str] = set()
    checksums: set[str] = set()
    total_bytes = 0
    count = 0
    if manifest.exists():
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                ids.add(record["stable_source_id"])
                checksums.add(record["content_checksum"])
                total_bytes += int(record["content_bytes"])
                count += 1
    return ids, checksums, total_bytes, count


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    for split in ("train", "validation"):
        (output / "documents" / split).mkdir(parents=True, exist_ok=True)
    provenance = output / "provenance.jsonl"
    known_ids, known_checksums, total_bytes, total_documents = existing_records(provenance)
    skipped: collections.Counter = collections.Counter()
    duplicates = 0
    retrieved = 0
    continuation: str | None = None
    retrieval_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    stop = False
    with provenance.open("a", encoding="utf-8") as manifest, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        while not stop:
            prefixes, continuation = list_prefixes(args.prefix, continuation, args.user_agent)
            if not prefixes:
                break
            metadata_results = pool.map(
                lambda value: fetch_metadata(value, args.user_agent), prefixes
            )
            accepted: list[tuple[str, dict]] = []
            for prefix, metadata, error in metadata_results:
                retrieved += 1
                if error or metadata is None:
                    skipped["metadata:" + str(error)] += 1
                    continue
                license_name = metadata.get("license_code")
                if license_name not in ALLOWLIST:
                    skipped["license:" + str(license_name)] += 1
                    continue
                if metadata.get("is_retracted") is True:
                    skipped["retracted"] += 1
                    continue
                source_id = str(metadata.get("pmcid") or "")
                if not source_id or source_id in known_ids:
                    skipped["known_or_missing_id"] += 1
                    continue
                accepted.append((prefix, metadata))
            article_results = pool.map(
                lambda value: fetch_article(value, args.user_agent), accepted
            )
            for prefix, metadata, root, error in article_results:
                if error or root is None:
                    skipped["xml:" + str(error)] += 1
                    continue
                article = next((item for item in root.iter() if local_name(item.tag) == "article"), root)
                content = extract_text(article)
                encoded = content.encode("utf-8")
                if len(encoded) < args.minimum_bytes:
                    skipped["too_short"] += 1
                    continue
                digest = hashlib.sha256(encoded).hexdigest()
                checksum = "sha256:" + digest
                source_id = str(metadata["pmcid"])
                if checksum in known_checksums:
                    duplicates += 1
                    known_ids.add(source_id)
                    continue
                split = (
                    "validation"
                    if int(digest[:16], 16) % 10_000 < args.validation_permyriad
                    else "train"
                )
                relative_path = pathlib.Path("documents") / split / f"{source_id}.txt"
                final_path = output / relative_path
                temporary_path = final_path.with_suffix(".txt.tmp")
                temporary_path.write_bytes(encoded)
                temporary_path.replace(final_path)
                record = {
                    "source_family": "PMC Cloud Service on AWS",
                    "stable_source_id": source_id,
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{source_id}/",
                    "official_identifier": source_id,
                    "pmc_version": metadata.get("version"),
                    "pmid": metadata.get("pmid"),
                    "doi": metadata.get("doi"),
                    "license": metadata["license_code"],
                    "license_evidence": "PMC machine-readable Cloud Service metadata",
                    "retrieval_date": retrieval_date,
                    "content_checksum": checksum,
                    "content_bytes": len(encoded),
                    "science_domain_category": subjects(article),
                    "split": split,
                    "document_path": str(final_path),
                    "cloud_prefix": prefix,
                }
                manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                manifest.flush()
                known_ids.add(source_id)
                known_checksums.add(checksum)
                total_bytes += len(encoded)
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
            if continuation is None:
                break
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
        description="Harvest allowlisted PMC JATS XML through the official public AWS dataset."
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--target-bytes", type=int, default=140_000_000)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--minimum-bytes", type=int, default=2_000)
    parser.add_argument("--validation-permyriad", type=int, default=100)
    parser.add_argument("--prefix", default="PMC100")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--user-agent", default="LedaCorpusResearch/0.1 (https://github.com/bzzling/leda)"
    )
    args = parser.parse_args()
    if args.target_bytes <= 0 or args.max_documents < 0 or args.minimum_bytes < 0:
        parser.error("byte/document limits must be nonnegative and target bytes must be positive")
    if not 0 < args.validation_permyriad < 10_000:
        parser.error("validation-permyriad must be in [1, 9999]")
    if not 1 <= args.workers <= 64:
        parser.error("workers must be in [1, 64]")
    run(args)


if __name__ == "__main__":
    main()
