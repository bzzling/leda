#!/usr/bin/env python3
"""Analyze Leda document-level validation and gradient-clipping metrics."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import random
import statistics


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def weighted_loss(rows: list[dict]) -> float:
    targets = sum(int(row["targets"]) for row in rows)
    return sum(int(row["targets"]) * float(row["mean_loss"]) for row in rows) / targets


def bootstrap_rows(rows: list[dict], seed: int, replicates: int) -> dict:
    generator = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        estimates.append(weighted_loss(sample))
    return {
        "documents": len(rows),
        "targets": sum(int(row["targets"]) for row in rows),
        "token_weighted_ce": weighted_loss(rows),
        "document_bootstrap_replicates": replicates,
        "document_bootstrap_95_low": percentile(estimates, 0.025),
        "document_bootstrap_95_high": percentile(estimates, 0.975),
    }


def bootstrap(args: argparse.Namespace) -> None:
    with args.document_losses.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("document loss CSV is empty")
    groups = {"global": rows}
    domains: dict[str, list[dict]] = collections.defaultdict(list)
    sources: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        domains[row["broad_domain"]].append(row)
        sources[row["source_family"]].append(row)
    groups.update({"domain:" + key: values for key, values in sorted(domains.items())})
    groups.update({"source:" + key: values for key, values in sorted(sources.items())})
    result = {
        key: bootstrap_rows(values, args.seed + index, args.replicates)
        for index, (key, values) in enumerate(groups.items())
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def summarize_gradients(rows: list[dict]) -> dict:
    gradients = [float(row["gradient_norm"]) for row in rows]
    return {
        "updates": len(rows),
        "start_step": int(rows[0]["step"]),
        "end_step": int(rows[-1]["step"]),
        "start_tokens": int(rows[0]["tokens_seen"]),
        "end_tokens": int(rows[-1]["tokens_seen"]),
        "clipping_frequency": statistics.fmean(int(row["clipped"]) for row in rows),
        "p50": percentile(gradients, 0.50),
        "p75": percentile(gradients, 0.75),
        "p90": percentile(gradients, 0.90),
        "p95": percentile(gradients, 0.95),
        "p99": percentile(gradients, 0.99),
        "max": max(gradients),
        "nonfinite_rows": sum(not math.isfinite(value) for value in gradients),
    }


def clipping(args: argparse.Namespace) -> None:
    with args.metrics.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) > 0]
    boundaries = []
    for item in args.blocks.split(","):
        name, separator, end = item.partition(":")
        if not separator or not name or not end.isdigit():
            raise ValueError("blocks must be comma-separated NAME:END_STEP values")
        boundaries.append((name, int(end)))
    if any(left[1] >= right[1] for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("block end steps must increase")
    result = {}
    start = 0
    for name, end in boundaries:
        block = [row for row in rows if start < int(row["step"]) <= end]
        if not block:
            raise ValueError(f"empty clipping block: {name}")
        result[name] = summarize_gradients(block)
        start = end
    if start < int(rows[-1]["step"]):
        block = [row for row in rows if int(row["step"]) > start]
        result["remainder"] = summarize_gradients(block)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = commands.add_parser("bootstrap")
    bootstrap_parser.add_argument("document_losses", type=pathlib.Path)
    bootstrap_parser.add_argument("--seed", type=int, default=3005)
    bootstrap_parser.add_argument("--replicates", type=int, default=10_000)
    bootstrap_parser.set_defaults(function=bootstrap)
    clipping_parser = commands.add_parser("clipping")
    clipping_parser.add_argument("metrics", type=pathlib.Path)
    clipping_parser.add_argument("--blocks", required=True)
    clipping_parser.set_defaults(function=clipping)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
