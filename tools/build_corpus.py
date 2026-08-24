#!/usr/bin/env python3
"""Deduplicate, split, sample, summarize, and fingerprint a Leda corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import pathlib
import re
import struct
import unicodedata
import zlib


WORD = re.compile(r"[^\W_]+", re.UNICODE)
SPARCORP_HEADER = struct.Struct("<8sIIQQIIQQ")


def normalize_identifier(value) -> str | None:
    if value is None:
        return None
    result = str(value).strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        result = result.removeprefix(prefix)
    return result or None


def document_path(record: dict, manifest: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record["local_document_path"])
    return path if path.is_absolute() else manifest.parent / path


def normalized_words(path: pathlib.Path) -> list[str]:
    text = unicodedata.normalize("NFKC", path.read_text(encoding="utf-8")).casefold()
    return WORD.findall(text)


def shingles(words: list[str], width: int = 5) -> set[int]:
    if len(words) < width:
        values = [" ".join(words)] if words else []
    else:
        values = (" ".join(words[index : index + width]) for index in range(len(words) - width + 1))
    return {int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "little") for value in values}


def bottom_shingle_sketch(words: list[str], width: int = 5, size: int = 16) -> list[int]:
    if len(words) < width:
        value = int.from_bytes(hashlib.blake2b(" ".join(words).encode(), digest_size=8).digest(), "little")
        return [value]
    mask = (1 << 64) - 1
    base = 1_099_511_628_211
    factor = pow(base, width, 1 << 64)
    codes = [zlib.crc32(word.encode()) + 1 for word in words]
    value = 0
    for code in codes[:width]:
        value = (value * base + code) & mask
    selected: set[int] = set()
    heap: list[int] = []

    def consider(candidate: int) -> None:
        if candidate in selected:
            return
        if len(heap) < size:
            heapq.heappush(heap, -candidate)
            selected.add(candidate)
        elif candidate < -heap[0]:
            selected.remove(-heapq.heapreplace(heap, -candidate))
            selected.add(candidate)

    consider(value)
    for index in range(width, len(codes)):
        value = (value * base + codes[index] - factor * codes[index - width]) & mask
        consider(value)
    return sorted(selected)


def jaccard(left: set[int], right: set[int]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def sketch_bands(sketch: list[int]) -> list[tuple[int, int]]:
    return [
        (sketch[index], sketch[index + 1] if index + 1 < len(sketch) else sketch[index])
        for index in range(0, len(sketch), 2)
    ]


def broad_domain(record: dict) -> str:
    if record.get("broad_domain"):
        return str(record["broad_domain"])
    evidence = " ".join(
        [str(record.get("title") or ""), str(record.get("source_category") or "")]
        + [str(item) for item in record.get("source_categories", [])]
    ).casefold()
    rules = [
        ("chemistry", ("chem", "molecule", "material science")),
        ("physics", ("physics", "astronom", "cosmolog", "geophys")),
        ("math_cs_engineering", ("mathemat", "computer", "algorithm", "engineering", "robot")),
        ("biology", ("ecology", "evolution", "plant", "zoolog", "microbiolog", "genetic")),
        ("explanatory_reference", ("education", "primer", "review", "encyclop", "textbook")),
        ("biomedical", ("medicine", "health", "clinical", "disease", "patient", "biomedical")),
    ]
    for name, markers in rules:
        if any(marker in evidence for marker in markers):
            return name
    return "biomedical" if record.get("source_family") in {"PMC Article Dataset", "PLOS"} else "other"


def read_candidates(manifests: list[pathlib.Path]) -> list[tuple[dict, pathlib.Path]]:
    result = []
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                required = {
                    "canonical_document_id",
                    "source_family",
                    "source_specific_id",
                    "license",
                    "license_evidence",
                    "retrieval_date",
                    "content_sha256",
                    "clean_text_bytes",
                    "local_document_path",
                }
                missing = required - record.keys()
                if missing:
                    raise ValueError(f"{manifest}:{line_number}: missing {sorted(missing)}")
                path = document_path(record, manifest)
                if not path.is_file():
                    raise ValueError(f"{manifest}:{line_number}: missing document {path}")
                if record["license"] not in {"CC0", "CC BY"}:
                    raise ValueError(f"{manifest}:{line_number}: license is not allowlisted")
                contents = path.read_bytes()
                if len(contents) != int(record["clean_text_bytes"]):
                    raise ValueError(f"{manifest}:{line_number}: clean-text byte count mismatch")
                if hashlib.sha256(contents).hexdigest() != record["content_sha256"]:
                    raise ValueError(f"{manifest}:{line_number}: clean-text SHA-256 mismatch")
                record["local_document_path"] = str(path.resolve())
                record["doi"] = normalize_identifier(record.get("doi"))
                if record.get("pmcid"):
                    record["pmcid"] = str(record["pmcid"]).upper()
                record["broad_domain"] = broad_domain(record)
                result.append((record, manifest))
    source_priority = {"PMC Article Dataset": 0, "PLOS": 1}
    result.sort(
        key=lambda item: (
            source_priority.get(item[0]["source_family"], 100),
            item[0]["canonical_document_id"],
        )
    )
    return result


def identifier_keys(record: dict) -> list[str]:
    values = [f"source:{record['source_family']}:{record['source_specific_id']}"]
    if record.get("pmcid"):
        values.append("pmcid:" + record["pmcid"])
    if record.get("doi"):
        values.append("doi:" + record["doi"])
    return values


def deduplicate(candidates: list[tuple[dict, pathlib.Path]], threshold: float) -> tuple[list[dict], dict]:
    unique: list[dict] = []
    identifiers: dict[str, int] = {}
    checksums: dict[str, int] = {}
    sketch_index: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    cached_shingles: dict[int, set[int]] = {}
    rejected = collections.Counter()
    duplicate_records = []
    for record, _ in candidates:
        reason = None
        survivor = None
        for key in identifier_keys(record):
            if key in identifiers:
                reason, survivor = "identifier_duplicate", identifiers[key]
                break
        if reason is None and record["content_sha256"] in checksums:
            reason, survivor = "exact_duplicate", checksums[record["content_sha256"]]
        sketch: list[int] | None = None
        if reason is None:
            words = normalized_words(pathlib.Path(record["local_document_path"]))
            sketch = bottom_shingle_sketch(words)
            possible: set[int] = set()
            for band in sketch_bands(sketch):
                possible.update(sketch_index.get(band, []))
            values: set[int] | None = None
            for index in sorted(possible):
                if values is None:
                    values = shingles(words)
                other = cached_shingles.get(index)
                if other is None:
                    other = shingles(normalized_words(pathlib.Path(unique[index]["local_document_path"])))
                    cached_shingles[index] = other
                    if len(cached_shingles) > 256:
                        cached_shingles.pop(next(iter(cached_shingles)))
                if jaccard(values, other) >= threshold:
                    reason, survivor = "near_duplicate", index
                    break
        if reason is not None:
            rejected[reason] += 1
            duplicate_records.append(
                {
                    "rejected_id": record["canonical_document_id"],
                    "survivor_id": unique[survivor]["canonical_document_id"],
                    "reason": reason,
                }
            )
            continue
        index = len(unique)
        unique.append(record)
        for key in identifier_keys(record):
            identifiers[key] = index
        checksums[record["content_sha256"]] = index
        assert sketch is not None
        for band in sketch_bands(sketch):
            sketch_index[band].append(index)
    return unique, {"counts": dict(sorted(rejected.items())), "records": duplicate_records}


def assign_splits(records: list[dict], seed: int, validation_permyriad: int, test_permyriad: int) -> None:
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for record in records:
        strata[(record["source_family"], record["broad_domain"])].append(record)
    for stratum, items in sorted(strata.items()):
        for record in items:
            key = f"{seed}:{stratum[0]}:{stratum[1]}:{record['canonical_document_id']}"
            value = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 10_000
            if value < test_permyriad:
                record["split"] = "test"
            elif value < test_permyriad + validation_permyriad:
                record["split"] = "validation"
            else:
                record["split"] = "train"


def stratified_sample(records: list[dict], maximum_bytes: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for record in records:
        if record["split"] == "train":
            groups[(record["source_family"], record["broad_domain"])].append(record)
    for key, values in groups.items():
        values.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}:{record['canonical_document_id']}".encode()
            ).digest()
        )
    total_training_bytes = sum(
        int(record["clean_text_bytes"]) for values in groups.values() for record in values
    )
    sample = []
    sampled_ids = set()
    used = 0
    for key, values in sorted(groups.items()):
        group_bytes = sum(int(record["clean_text_bytes"]) for record in values)
        budget = math.floor(maximum_bytes * group_bytes / total_training_bytes)
        group_used = 0
        for record in values:
            size = int(record["clean_text_bytes"])
            if group_used and size > budget - group_used:
                continue
            if used and size > maximum_bytes - used:
                break
            sample.append(record)
            sampled_ids.add(record["canonical_document_id"])
            used += size
            group_used += size
            if group_used >= budget:
                break
    remaining = sorted(
        (record for values in groups.values() for record in values if record["canonical_document_id"] not in sampled_ids),
        key=lambda record: hashlib.sha256(
            f"{seed}:remainder:{record['canonical_document_id']}".encode()
        ).digest(),
    )
    for record in remaining:
        size = int(record["clean_text_bytes"])
        if size <= maximum_bytes - used:
            sample.append(record)
            used += size
        if used >= maximum_bytes:
            break
    return sample


def write_manifest(path: pathlib.Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in sorted(records, key=lambda item: item["canonical_document_id"]):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "unknown"


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = read_candidates(args.candidate_manifests)
    unique, duplicates = deduplicate(candidates, args.near_duplicate_threshold)
    assign_splits(unique, args.split_seed, args.validation_permyriad, args.test_permyriad)
    write_manifest(output / "master-manifest.jsonl", unique)
    with (output / "duplicates.jsonl").open("w", encoding="utf-8") as stream:
        for record in sorted(duplicates["records"], key=lambda item: item["rejected_id"]):
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    for split in ("train", "validation", "test"):
        selected = [record for record in unique if record["split"] == split]
        (output / f"{split}.paths").write_text(
            "".join(record["local_document_path"] + "\n" for record in sorted(selected, key=lambda item: item["canonical_document_id"])),
            encoding="utf-8",
        )
    views = output / "validation-views"
    views.mkdir(exist_ok=True)
    validation_records = [record for record in unique if record["split"] == "validation"]
    with (output / "validation-documents.tsv").open("w", encoding="utf-8") as stream:
        stream.write(
            "canonical_document_id\tbroad_domain\tsource_family\tlocal_document_path\n"
        )
        for record in sorted(validation_records, key=lambda item: item["canonical_document_id"]):
            values = [
                str(record["canonical_document_id"]),
                str(record["broad_domain"]),
                str(record["source_family"]),
                str(record["local_document_path"]),
            ]
            if any("\t" in value or "\n" in value for value in values):
                raise ValueError("Validation TSV fields may not contain tabs or newlines")
            stream.write("\t".join(values) + "\n")
    with (output / "mixture-documents.tsv").open("w", encoding="utf-8") as stream:
        stream.write(
            "split\tsource_family\tbroad_domain\tlicense\tlocal_document_path\n"
        )
        for record in sorted(unique, key=lambda item: item["canonical_document_id"]):
            values = [
                str(record["split"]),
                str(record["source_family"]),
                str(record["broad_domain"]),
                str(record["license"]),
                str(record["local_document_path"]),
            ]
            if any("\t" in value or "\n" in value for value in values):
                raise ValueError("Mixture TSV fields may not contain tabs or newlines")
            stream.write("\t".join(values) + "\n")
    for field in ("broad_domain", "source_family"):
        grouped: dict[str, list[dict]] = collections.defaultdict(list)
        for record in validation_records:
            grouped[str(record[field])].append(record)
        for value, records in grouped.items():
            (views / f"{field}-{safe_name(value)}.paths").write_text(
                "".join(
                    record["local_document_path"] + "\n"
                    for record in sorted(records, key=lambda item: item["canonical_document_id"])
                ),
                encoding="utf-8",
            )
    mixture_views = output / "mixture-views"
    mixture_views.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [record for record in unique if record["split"] == split]
        for field in ("broad_domain", "source_family", "license"):
            grouped: dict[str, list[dict]] = collections.defaultdict(list)
            for record in selected:
                grouped[str(record[field])].append(record)
            for value, records in grouped.items():
                (mixture_views / f"{split}-{field}-{safe_name(value)}.paths").write_text(
                    "".join(
                        record["local_document_path"] + "\n"
                        for record in sorted(
                            records, key=lambda item: item["canonical_document_id"]
                        )
                    ),
                    encoding="utf-8",
                )
    sample = stratified_sample(unique, args.tokenizer_sample_bytes, args.tokenizer_seed)
    write_manifest(output / "tokenizer-sample.jsonl", sample)
    (output / "tokenizer-sample.paths").write_text(
        "".join(record["local_document_path"] + "\n" for record in sample), encoding="utf-8"
    )
    by_split = collections.Counter(record["split"] for record in unique)
    bytes_by_split = collections.Counter()
    by_source = collections.Counter()
    by_domain = collections.Counter()
    by_license = collections.Counter()
    for record in unique:
        size = int(record["clean_text_bytes"])
        bytes_by_split[record["split"]] += size
        by_source[record["source_family"]] += size
        by_domain[record["broad_domain"]] += size
        by_license[record["license"]] += size
    sample_composition = collections.Counter(
        (record["source_family"], record["broad_domain"]) for record in sample
    )
    sample_bytes_composition = collections.Counter()
    for record in sample:
        sample_bytes_composition[(record["source_family"], record["broad_domain"])] += int(
            record["clean_text_bytes"]
        )
    source_recipe = None
    source_recipe_sha256 = None
    if args.source_recipe:
        source_recipe_contents = args.source_recipe.read_bytes()
        source_recipe = json.loads(source_recipe_contents)
        if not isinstance(source_recipe, dict):
            raise ValueError("source recipe must be a JSON object")
        source_recipe_sha256 = hashlib.sha256(source_recipe_contents).hexdigest()
    recipe = {
        "format": "LEDA_CORPUS_RECIPE_V1",
        "candidate_manifests": [
            {"path": str(path), "sha256": sha256(path)}
            for path in args.candidate_manifests
        ],
        "license_allowlist": ["CC0", "CC BY"],
        "near_duplicate_method": (
            "exact word-5-shingle Jaccard after 8 paired bands from a 16-value bottom-shingle sketch"
        ),
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "split_seed": args.split_seed,
        "validation_permyriad": args.validation_permyriad,
        "test_permyriad": args.test_permyriad,
        "tokenizer_seed": args.tokenizer_seed,
        "tokenizer_sample_target_bytes": args.tokenizer_sample_bytes,
        "source_recipe": source_recipe,
        "source_recipe_sha256": source_recipe_sha256,
    }
    (output / "corpus-recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    summary = {
        "raw_candidate_documents": len(candidates),
        "admitted_documents": len(unique),
        "duplicate_counts": duplicates["counts"],
        "documents_by_split": dict(sorted(by_split.items())),
        "bytes_by_split": dict(sorted(bytes_by_split.items())),
        "bytes_by_source": dict(sorted(by_source.items())),
        "bytes_by_domain": dict(sorted(by_domain.items())),
        "bytes_by_license": dict(sorted(by_license.items())),
        "tokenizer_sample_documents": len(sample),
        "tokenizer_sample_bytes": sum(int(record["clean_text_bytes"]) for record in sample),
        "tokenizer_sample_composition": {
            f"{key[0]}::{key[1]}": value for key, value in sorted(sample_composition.items())
        },
        "tokenizer_sample_bytes_composition": {
            f"{key[0]}::{key[1]}": value
            for key, value in sorted(sample_bytes_composition.items())
        },
    }
    (output / "build-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def shard_metadata(path: pathlib.Path, root: pathlib.Path) -> dict:
    resolved = path.resolve()
    try:
        relative_path = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"SPARCORP shard is outside the portable corpus root: {path}") from error
    with resolved.open("rb") as stream:
        header = stream.read(SPARCORP_HEADER.size)
    if len(header) != SPARCORP_HEADER.size:
        raise ValueError(f"Truncated SPARCORP header: {path}")
    magic, version, header_size, tokenizer_vocab, model_vocab, eod, reserved, tokens, documents = SPARCORP_HEADER.unpack(header)
    if magic != b"SPARCORP" or version != 1 or header_size != SPARCORP_HEADER.size or reserved != 0:
        raise ValueError(f"Invalid SPARCORP v1 header: {path}")
    return {
        "relative_path": relative_path,
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
        "tokens": tokens,
        "documents": documents,
        "tokenizer_vocab": tokenizer_vocab,
        "model_vocab": model_vocab,
        "eod": eod,
    }


def freeze(args: argparse.Namespace) -> None:
    root = args.output.resolve().parent
    records = []
    with args.master_manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    shards = {
        split: [shard_metadata(path, root) for path in paths]
        for split, paths in args.shards.items()
    }
    documents_by_split = collections.Counter(record.get("split") for record in records)
    if set(documents_by_split) != {"train", "validation", "test"}:
        raise ValueError("master manifest must contain all three recognized splits")
    for split, values in shards.items():
        shard_documents = sum(int(value["documents"]) for value in values)
        if shard_documents != documents_by_split[split]:
            raise ValueError(f"{split} shard document count conflicts with master manifest")
    header_identities = {
        (value["tokenizer_vocab"], value["model_vocab"], value["eod"])
        for values in shards.values()
        for value in values
    }
    if len(header_identities) != 1:
        raise ValueError("SPARCORP shard vocabulary metadata is inconsistent")
    tokenizer_hash = sha256(args.tokenizer)
    identity = hashlib.sha256()
    identity.update(b"LEDA_CORPUS_V1\n")
    identity.update(f"tokenizer:{tokenizer_hash}\n".encode())
    for record in sorted(records, key=lambda item: item["canonical_document_id"]):
        identity.update(
            f"document:{record['canonical_document_id']}:{record['content_sha256']}:{record['split']}\n".encode()
        )
    for split in sorted(shards):
        for shard in sorted(shards[split], key=lambda item: item["relative_path"]):
            identity.update(
                f"shard:{split}:{shard['relative_path']}:{shard['sha256']}\n".encode()
            )
    result = {
        "format": "LEDA_CORPUS_FREEZE_V1",
        "corpus_fingerprint": identity.hexdigest(),
        "tokenizer_sha256": tokenizer_hash,
        "documents": len(records),
        "documents_by_split": dict(sorted(documents_by_split.items())),
        "tokens_by_split": {
            split: sum(int(value["tokens"]) for value in values)
            for split, values in sorted(shards.items())
        },
        "bytes_by_split": {
            split: sum(int(value["bytes"]) for value in values)
            for split, values in sorted(shards.items())
        },
        "shards": shards,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    freeze_record = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze_record.get("format") != "LEDA_CORPUS_FREEZE_V1":
        raise ValueError("unsupported corpus freeze format")
    tokenizer_hash = sha256(args.tokenizer)
    if tokenizer_hash != freeze_record.get("tokenizer_sha256"):
        raise ValueError("tokenizer SHA-256 conflicts with corpus freeze")
    root = args.freeze.resolve().parent
    shards = freeze_record.get("shards")
    if not isinstance(shards, dict):
        raise ValueError("corpus freeze has no shard map")
    for split, values in shards.items():
        for value in values:
            path = root / value["relative_path"]
            if shard_metadata(path, root) != value:
                raise ValueError(f"{split} SPARCORP shard conflicts with corpus freeze: {path}")
    with args.master_manifest.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    identity = hashlib.sha256()
    identity.update(b"LEDA_CORPUS_V1\n")
    identity.update(f"tokenizer:{tokenizer_hash}\n".encode())
    for record in sorted(records, key=lambda item: item["canonical_document_id"]):
        identity.update(
            f"document:{record['canonical_document_id']}:{record['content_sha256']}:{record['split']}\n".encode()
        )
    for split in sorted(shards):
        for shard in sorted(shards[split], key=lambda item: item["relative_path"]):
            identity.update(
                f"shard:{split}:{shard['relative_path']}:{shard['sha256']}\n".encode()
            )
    if identity.hexdigest() != freeze_record.get("corpus_fingerprint"):
        raise ValueError("master manifest identity conflicts with corpus freeze")
    print(
        json.dumps(
            {
                "corpus_fingerprint": identity.hexdigest(),
                "tokenizer_sha256": tokenizer_hash,
                "documents": len(records),
                "shards": sum(len(values) for values in shards.values()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("output", type=pathlib.Path)
    build_parser.add_argument("candidate_manifests", type=pathlib.Path, nargs="+")
    build_parser.add_argument("--source-recipe", type=pathlib.Path)
    build_parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    build_parser.add_argument("--split-seed", type=int, default=3002)
    build_parser.add_argument("--validation-permyriad", type=int, default=100)
    build_parser.add_argument("--test-permyriad", type=int, default=100)
    build_parser.add_argument("--tokenizer-seed", type=int, default=3003)
    build_parser.add_argument("--tokenizer-sample-bytes", type=int, default=134_217_728)
    build_parser.set_defaults(function=build)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("master_manifest", type=pathlib.Path)
    freeze_parser.add_argument("tokenizer", type=pathlib.Path)
    freeze_parser.add_argument("output", type=pathlib.Path)
    freeze_parser.add_argument("--train-shards", type=pathlib.Path, nargs="+", required=True)
    freeze_parser.add_argument("--validation-shards", type=pathlib.Path, nargs="+", required=True)
    freeze_parser.add_argument("--test-shards", type=pathlib.Path, nargs="+", required=True)
    freeze_parser.set_defaults(function=freeze)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("freeze", type=pathlib.Path)
    verify_parser.add_argument("master_manifest", type=pathlib.Path)
    verify_parser.add_argument("tokenizer", type=pathlib.Path)
    verify_parser.set_defaults(function=verify)
    args = parser.parse_args()
    if args.command == "build":
        if not 0.0 < args.near_duplicate_threshold <= 1.0:
            parser.error("near-duplicate threshold must be in (0,1]")
        if args.validation_permyriad <= 0 or args.test_permyriad <= 0 or args.validation_permyriad + args.test_permyriad >= 10_000:
            parser.error("split rates must be positive and leave training data")
    elif args.command == "freeze":
        args.shards = {
            "train": args.train_shards,
            "validation": args.validation_shards,
            "test": args.test_shards,
        }
    args.function(args)


if __name__ == "__main__":
    main()
