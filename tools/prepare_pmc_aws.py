#!/usr/bin/env python3
"""Audit and acquire allowlisted PMC articles through the official daily S3 inventory."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import gzip
import hashlib
import json
import pathlib
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from prepare_pmc import extract_text, local_name, subjects


BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
INVENTORY_ROOT = "inventory-reports/pmc-oa-opendata/metadata"
ALLOWLIST = {"CC0", "CC BY"}
EXCLUDED_ARTICLE_TYPES = {
    "addendum",
    "correction",
    "corrigendum",
    "erratum",
    "expression-of-concern",
    "retraction",
    "retraction-notice",
}


def request(url: str, user_agent: str, *, method: str = "GET", attempts: int = 5):
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": user_agent}, method=method),
                timeout=180,
            )
        except (OSError, urllib.error.HTTPError) as caught:
            error = caught
            if attempt + 1 != attempts:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def get_bytes(url: str, user_agent: str) -> tuple[bytes, dict[str, str]]:
    with request(url, user_agent) as response:
        return response.read(), {key.casefold(): value for key, value in response.headers.items()}


def current_manifest(user_agent: str) -> tuple[str, bytes]:
    today = dt.datetime.now(dt.timezone.utc).date()
    for offset in range(8):
        day = today - dt.timedelta(days=offset)
        key = f"{INVENTORY_ROOT}/{day.isoformat()}T01-00Z/manifest.json"
        try:
            contents, _ = get_bytes(BUCKET + key, user_agent)
            return key, contents
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    raise RuntimeError("No PMC metadata inventory manifest was published in the last eight days")


def md5(path: pathlib.Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: pathlib.Path, expected_md5: str, user_agent: str) -> None:
    if destination.exists() and md5(destination) == expected_md5:
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    contents, _ = get_bytes(url, user_agent)
    temporary.write_bytes(contents)
    if md5(temporary) != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Inventory checksum mismatch: {url}")
    temporary.replace(destination)


def initialize(database: sqlite3.Connection) -> None:
    database.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS versions(
          pmcid TEXT NOT NULL,
          version INTEGER NOT NULL,
          object_key TEXT NOT NULL UNIQUE,
          last_modified TEXT NOT NULL,
          etag TEXT NOT NULL,
          sample_key TEXT NOT NULL,
          PRIMARY KEY(pmcid, version)
        );
        CREATE INDEX IF NOT EXISTS versions_sample ON versions(sample_key, pmcid, version DESC);
        """
    )


def parse_metadata_key(key: str) -> tuple[str, int] | None:
    if not key.startswith("metadata/PMC") or not key.endswith(".json"):
        return None
    stem = key.removeprefix("metadata/").removesuffix(".json")
    pmcid, separator, version = stem.rpartition(".")
    if not separator or not pmcid.removeprefix("PMC").isdigit() or not version.isdigit():
        return None
    return pmcid, int(version)


def index_inventory(output: pathlib.Path, manifest_url: str | None, seed: int, user_agent: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    inventory_dir = output / "inventory"
    inventory_dir.mkdir(exist_ok=True)
    if manifest_url:
        manifest_contents, _ = get_bytes(manifest_url, user_agent)
        manifest_key = manifest_url.removeprefix(BUCKET)
    else:
        manifest_key, manifest_contents = current_manifest(user_agent)
        manifest_url = BUCKET + manifest_key
    manifest = json.loads(manifest_contents)
    if manifest.get("fileFormat") != "CSV" or manifest.get("fileSchema") != "Bucket, Key, LastModifiedDate, ETag":
        raise RuntimeError("Unsupported PMC inventory schema")
    (output / "inventory-manifest.json").write_bytes(manifest_contents)
    (output / "inventory-source.json").write_text(
        json.dumps({"url": manifest_url, "sha256": hashlib.sha256(manifest_contents).hexdigest()}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_contents).hexdigest()
    index_identity = {
        "format": "LEDA_PMC_INVENTORY_INDEX_V1",
        "manifest_sha256": manifest_sha256,
        "seed": seed,
    }
    state_path = output / "inventory-index-state.json"
    database = sqlite3.connect(output / "inventory.sqlite")
    initialize(database)
    cached_state = json.loads(state_path.read_text()) if state_path.exists() else None
    if cached_state == index_identity:
        counts = database.execute(
            "SELECT COUNT(*), COUNT(DISTINCT pmcid), MAX(version) FROM versions"
        ).fetchone()
        database.close()
        return {
            "inventory_manifest_url": manifest_url,
            "inventory_manifest_sha256": manifest_sha256,
            "inventory_rows_read": counts[0],
            "version_objects": counts[0],
            "unique_pmcids": counts[1],
            "maximum_version": counts[2],
            "inventory_index_reused": True,
        }
    database.execute("DELETE FROM versions")
    database.commit()
    rows = 0
    for item in manifest["files"]:
        key = item["key"]
        path = inventory_dir / pathlib.Path(key).name
        download(BUCKET + key, path, item["MD5checksum"], user_agent)
        batch = []
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for bucket, object_key, modified, etag in csv.reader(stream):
                if bucket != "pmc-oa-opendata":
                    raise RuntimeError(f"Unexpected inventory bucket: {bucket}")
                parsed = parse_metadata_key(object_key)
                if parsed is None:
                    continue
                pmcid, version = parsed
                sample_key = hashlib.sha256(f"{seed}:{pmcid}".encode()).hexdigest()
                batch.append((pmcid, version, object_key, modified, etag.strip('"'), sample_key))
                if len(batch) == 10_000:
                    database.executemany("INSERT OR REPLACE INTO versions VALUES(?,?,?,?,?,?)", batch)
                    rows += len(batch)
                    batch.clear()
            database.executemany("INSERT OR REPLACE INTO versions VALUES(?,?,?,?,?,?)", batch)
            rows += len(batch)
            database.commit()
    counts = database.execute(
        "SELECT COUNT(*), COUNT(DISTINCT pmcid), MAX(version) FROM versions"
    ).fetchone()
    database.close()
    state_path.write_text(json.dumps(index_identity, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "inventory_manifest_url": manifest_url,
        "inventory_manifest_sha256": manifest_sha256,
        "inventory_rows_read": rows,
        "version_objects": counts[0],
        "unique_pmcids": counts[1],
        "maximum_version": counts[2],
        "inventory_index_reused": False,
    }


def version_groups(database: sqlite3.Connection, limit: int) -> list[list[tuple]]:
    groups: list[list[tuple]] = []
    current: list[tuple] = []
    previous = None
    query = "SELECT pmcid,version,object_key,last_modified,etag FROM versions ORDER BY sample_key,pmcid,version DESC"
    for row in database.execute(query):
        if previous is not None and row[0] != previous:
            groups.append(current)
            current = []
            if limit and len(groups) >= limit:
                break
        current.append(row)
        previous = row[0]
    if current and (not limit or len(groups) < limit):
        groups.append(current)
    return groups


def object_from_s3(url: str) -> tuple[str, str | None]:
    if not url.startswith("s3://pmc-oa-opendata/"):
        raise ValueError("PMC content URL is outside the official bucket")
    path, _, query = url.removeprefix("s3://pmc-oa-opendata/").partition("?")
    checksum = urllib.parse.parse_qs(query).get("md5", [None])[0]
    return path, checksum


def quality_reason(text: str, minimum_bytes: int) -> str | None:
    encoded = text.encode("utf-8")
    if len(encoded) < minimum_bytes:
        return "too_short"
    if not text.strip():
        return "extraction_empty"
    controls = sum(ord(character) < 32 and character not in "\n\t\r" for character in text)
    if controls / max(1, len(text)) > 0.001:
        return "pathological_character_ratio"
    return None


def retrieve_group(group: list[tuple], user_agent: str, minimum_bytes: int) -> tuple[dict | None, str]:
    for pmcid, version, metadata_key, modified, metadata_etag in group:
        try:
            raw_metadata, _ = get_bytes(BUCKET + urllib.parse.quote(metadata_key), user_agent)
            metadata = json.loads(raw_metadata)
        except (OSError, ValueError) as error:
            return None, "metadata:" + type(error).__name__
        if metadata.get("is_retracted") is True:
            return None, "retracted"
        if metadata.get("license_code") not in ALLOWLIST:
            continue
        xml_url = metadata.get("xml_url")
        if not isinstance(xml_url, str):
            return None, "missing_xml"
        try:
            content_key, expected_md5 = object_from_s3(xml_url)
            raw_xml, headers = get_bytes(BUCKET + urllib.parse.quote(content_key), user_agent)
            if expected_md5 and hashlib.md5(raw_xml, usedforsecurity=False).hexdigest() != expected_md5:
                return None, "content_md5_mismatch"
            root = ET.fromstring(raw_xml)
        except (OSError, ValueError, ET.ParseError) as error:
            return None, "xml:" + type(error).__name__
        article = next((item for item in root.iter() if local_name(item.tag) == "article"), root)
        article_type = str(article.attrib.get("article-type") or "unknown").casefold()
        if article_type in EXCLUDED_ARTICLE_TYPES:
            return None, "article_type:" + article_type
        text = extract_text(article)
        reason = quality_reason(text, minimum_bytes)
        if reason:
            return None, reason
        encoded = text.encode("utf-8")
        record = {
            "canonical_document_id": f"pmc:{pmcid}.{version}",
            "source_family": "PMC Article Dataset",
            "source_specific_id": pmcid,
            "pmcid": pmcid,
            "doi": metadata.get("doi"),
            "pmid": metadata.get("pmid"),
            "article_version": version,
            "title": metadata.get("title"),
            "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
            "license": metadata["license_code"],
            "license_evidence": "PMC Article Dataset machine-readable metadata license_code",
            "retrieval_date": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "content_sha256": hashlib.sha256(encoded).hexdigest(),
            "raw_metadata_sha256": hashlib.sha256(raw_metadata).hexdigest(),
            "raw_metadata_etag": metadata_etag,
            "raw_metadata_last_modified": modified,
            "source_content_key": content_key,
            "source_content_etag": headers.get("etag", "").strip('"'),
            "source_content_md5": expected_md5,
            "clean_text_bytes": len(encoded),
            "source_categories": subjects(article),
            "article_type": article_type,
            "text": text,
        }
        return record, "admitted"
    return None, "no_eligible_version"


def collect(args: argparse.Namespace, *, audit: bool) -> dict:
    output = args.output.resolve()
    index_summary = index_inventory(output, args.inventory_manifest_url, args.seed, args.user_agent)
    database = sqlite3.connect(output / "inventory.sqlite")
    groups = version_groups(database, args.sample_documents if audit else args.maximum_documents)
    database.close()
    records: list[dict] = []
    rejected: collections.Counter[str] = collections.Counter()
    total_bytes = 0
    groups_examined = 0
    documents_dir = output / "documents"
    if not audit:
        documents_dir.mkdir(exist_ok=True)
    fetch_batch_size = max(32, args.workers * 4)
    target_reached = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(groups), fetch_batch_size):
            batch = groups[offset : offset + fetch_batch_size]
            results = pool.map(
                lambda group: retrieve_group(group, args.user_agent, args.minimum_bytes), batch
            )
            for record, reason in results:
                groups_examined += 1
                if record is None:
                    rejected[reason] += 1
                    continue
                text = record.pop("text")
                total_bytes += record["clean_text_bytes"]
                if not audit:
                    relative = pathlib.Path("documents") / (
                        f"{record['pmcid']}.{record['article_version']}.txt"
                    )
                    (output / relative).write_text(text, encoding="utf-8")
                    record["local_document_path"] = relative.as_posix()
                records.append(record)
                if not audit and args.target_bytes and total_bytes >= args.target_bytes:
                    target_reached = True
                    break
            if target_reached:
                break
    if not audit:
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        with (output / "audit-sample.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    licenses = collections.Counter(record["license"] for record in records)
    categories = collections.Counter(
        category for record in records for category in record.get("source_categories", [])
    )
    article_types = collections.Counter(record.get("article_type") or "unknown" for record in records)
    summary = {
        **index_summary,
        "mode": "audit" if audit else "acquire",
        "groups_selected": len(groups),
        "groups_examined": groups_examined,
        "admitted_documents": len(records),
        "clean_text_bytes": total_bytes,
        "mean_clean_bytes": total_bytes / len(records) if records else 0,
        "estimated_tokens_at_4_bytes_per_token": total_bytes // 4,
        "licenses": dict(sorted(licenses.items())),
        "top_source_categories": dict(categories.most_common(50)),
        "article_types": dict(sorted(article_types.items())),
        "rejections": dict(sorted(rejected.items())),
    }
    (output / ("audit-summary.json" if audit else "acquisition-summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "acquire"):
        command = commands.add_parser(name)
        command.add_argument("output", type=pathlib.Path)
        command.add_argument("--inventory-manifest-url")
        command.add_argument("--seed", type=int, default=3001)
        command.add_argument("--workers", type=int, default=16)
        command.add_argument("--minimum-bytes", type=int, default=2_000)
        command.add_argument("--sample-documents", type=int, default=1_000)
        command.add_argument("--maximum-documents", type=int, default=0)
        command.add_argument("--target-bytes", type=int, default=0)
        command.add_argument(
            "--user-agent", default="LedaCorpusResearch/0.1 (https://github.com/bzzling/leda)"
        )
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or args.minimum_bytes < 0 or args.seed < 0:
        parser.error("invalid worker, minimum-byte, or seed setting")
    if args.command == "audit" and args.sample_documents <= 0:
        parser.error("audit sample must be positive")
    if args.command == "acquire" and args.maximum_documents <= 0:
        parser.error("acquire requires a positive maximum document scan bound")
    print(json.dumps(collect(args, audit=args.command == "audit"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
