#!/usr/bin/env python3
"""Summarize Leda document validation and fixed-prefix diagnostics."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import statistics


def weighted_ce(rows: list[dict[str, str]]) -> float:
    targets = sum(int(row["targets"]) for row in rows)
    if targets == 0:
        raise ValueError("evaluation rows contain no targets")
    return sum(int(row["targets"]) * float(row["mean_loss"]) for row in rows) / targets


def bootstrap(rows: list[dict[str, str]], samples: int, seed: int) -> dict[str, float | int]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [rows[generator.randrange(len(rows))] for _ in rows]
        estimates.append(weighted_ce(draw))
    estimates.sort()

    def quantile(fraction: float) -> float:
        position = fraction * (len(estimates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (position - lower)

    return {
        "documents": len(rows),
        "targets": sum(int(row["targets"]) for row in rows),
        "token_weighted_ce": weighted_ce(rows),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bootstrap_95_low": quantile(0.025),
        "bootstrap_95_high": quantile(0.975),
    }


def documents(args: argparse.Namespace) -> None:
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {
        "canonical_document_id",
        "broad_domain",
        "source_family",
        "targets",
        "mean_loss",
    }:
        raise ValueError("unexpected document-evaluation CSV schema")
    by_domain: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_domain[row["broad_domain"]].append(row)
    result = {
        "global": bootstrap(rows, args.samples, args.seed),
        "domains": {
            name: bootstrap(group, args.samples, args.seed + index + 1)
            for index, (name, group) in enumerate(sorted(by_domain.items()))
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def prefixes(args: argparse.Namespace) -> None:
    stem_counts = collections.Counter(path.stem for path in args.inputs)
    inputs: list[tuple[str, list[dict[str, str]]]] = []
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"empty prefix evaluation: {path}")
        name = path.stem
        if stem_counts[name] > 1:
            name = f"{path.parent.name}/{name}"
        inputs.append((name, rows))
    expected = [
        (row["prefix_group"], row["canonical_document_id"], row["broad_domain"])
        for row in inputs[0][1]
    ]
    result = {}
    for name, rows in inputs:
        identity = [
            (row["prefix_group"], row["canonical_document_id"], row["broad_domain"])
            for row in rows
        ]
        if identity != expected:
            raise ValueError("prefix inputs do not contain the same fixed examples in the same order")
        grouped: dict[str, list[float]] = collections.defaultdict(list)
        for row in rows:
            grouped[row["prefix_group"]].append(float(row["mean_loss"]))
        result[name] = {
            prefix: {
                "examples": len(losses),
                "continuation_ce": statistics.fmean(losses),
            }
            for prefix, losses in sorted(grouped.items(), key=lambda item: int(item[0]))
        }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    document_parser = commands.add_parser("documents")
    document_parser.add_argument("input", type=pathlib.Path)
    document_parser.add_argument("--samples", type=int, default=10_000)
    document_parser.add_argument("--seed", type=int, default=3100)
    document_parser.set_defaults(function=documents)
    prefix_parser = commands.add_parser("prefixes")
    prefix_parser.add_argument("inputs", type=pathlib.Path, nargs="+")
    prefix_parser.set_defaults(function=prefixes)
    args = parser.parse_args()
    if getattr(args, "samples", 1) <= 0:
        raise ValueError("bootstrap sample count must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
