#!/usr/bin/env python3
"""Analyze and freeze the deterministic Leda scientific-autocomplete evaluation."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
from typing import Any


def rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"empty study input: {path}")
    return result


def estimate(group: list[dict[str, str]]) -> dict[str, float | int]:
    targets = sum(int(row["targets"]) for row in group)
    loss_sum = sum(float(row["loss_sum"]) for row in group)
    if targets == 0:
        raise ValueError("metric group has no targets")
    ce = loss_sum / targets
    return {"documents": len(group), "targets": targets, "ce": ce, "perplexity": math.exp(ce)}


def bootstrap(
    group: list[dict[str, str]], samples: int, seed: int
) -> dict[str, float | int]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [group[generator.randrange(len(group))] for _ in group]
        estimates.append(float(estimate(draw)["ce"]))
    estimates.sort()

    def quantile(fraction: float) -> float:
        position = fraction * (len(estimates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        return estimates[lower] + (estimates[upper] - estimates[lower]) * (position - lower)

    result = estimate(group)
    low = quantile(0.025)
    high = quantile(0.975)
    result.update(
        {
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "ce_95_low": low,
            "ce_95_high": high,
            "perplexity_95_low": math.exp(low),
            "perplexity_95_high": math.exp(high),
        }
    )
    return result


def next_metrics(group: list[dict[str, str]]) -> dict[str, float | int]:
    positions = sum(int(row["positions"]) for row in group)
    if positions == 0:
        raise ValueError("next-token group has no positions")
    return {
        "positions": positions,
        "top1_accuracy": sum(int(row["top1"]) for row in group) / positions,
        "top5_accuracy": sum(int(row["top5"]) for row in group) / positions,
        "top10_accuracy": sum(int(row["top10"]) for row in group) / positions,
        "mean_correct_probability": sum(
            float(row["correct_probability_sum"]) for row in group
        )
        / positions,
        "mean_entropy_nats": sum(float(row["entropy_sum"]) for row in group) / positions,
    }


def analyze_metrics(args: argparse.Namespace) -> None:
    documents = rows(args.directory / "documents.csv")
    next_rows = rows(args.directory / "next-token.csv")
    prefix_rows = rows(args.directory / "prefixes.csv")
    by_domain: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in documents:
        by_domain[row["broad_domain"]].append(row)
    next_by_domain: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in next_rows:
        next_by_domain[row["broad_domain"]].append(row)

    primary: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    ablation: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in prefix_rows:
        key = (row["prefix_tokens"], row["available_context"])
        ablation[key].append(
            {
                "targets": row["continuation_targets"],
                "loss_sum": row["loss_sum"],
            }
        )
        if row["prefix_tokens"] == row["available_context"]:
            primary[row["prefix_tokens"]].append(
                {
                    "targets": row["continuation_targets"],
                    "loss_sum": row["loss_sum"],
                }
            )
    result = {
        "split": documents[0]["split"],
        "language_model": {
            "global": bootstrap(documents, args.bootstrap_samples, args.bootstrap_seed),
            "domains": {
                domain: bootstrap(group, args.bootstrap_samples, args.bootstrap_seed + index + 1)
                for index, (domain, group) in enumerate(sorted(by_domain.items()))
            },
        },
        "next_token": {
            "global": next_metrics(next_rows),
            "domains": {
                domain: next_metrics(group) for domain, group in sorted(next_by_domain.items())
            },
        },
        "prefix_continuation": {
            prefix: estimate(group)
            for prefix, group in sorted(primary.items(), key=lambda item: int(item[0]))
        },
        "context_ablation": {
            f"prefix_{prefix}_context_{context}": estimate(group)
            for (prefix, context), group in sorted(
                ablation.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
            )
            if prefix in {"256", "400"}
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


def token_ids(value: str) -> list[int]:
    return [] if not value else [int(token) for token in value.split()]


def repeated_fraction(tokens: list[int], width: int) -> float:
    if len(tokens) < width:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeated = 0
    for index in range(len(tokens) - width + 1):
        gram = tuple(tokens[index : index + width])
        repeated += gram in seen
        seen.add(gram)
    return repeated / (len(tokens) - width + 1)


def longest_repeat(tokens: list[int]) -> int:
    longest = 0
    for left in range(len(tokens)):
        for right in range(left + 1, len(tokens)):
            length = 0
            while right + length < len(tokens) and tokens[left + length] == tokens[right + length]:
                length += 1
            longest = max(longest, length)
    return longest


def invalid_utf8_bytes(data: bytes) -> int:
    invalid = 0
    index = 0
    while index < len(data):
        first = data[index]
        if first < 0x80:
            index += 1
            continue
        width = 2 if 0xC2 <= first <= 0xDF else 3 if 0xE0 <= first <= 0xEF else 4 if 0xF0 <= first <= 0xF4 else 0
        valid = width > 0 and index + width <= len(data) and all(
            data[index + offset] & 0xC0 == 0x80 for offset in range(1, width)
        )
        if valid and width == 3:
            valid = not ((first == 0xE0 and data[index + 1] < 0xA0) or (first == 0xED and data[index + 1] >= 0xA0))
        if valid and width == 4:
            valid = not ((first == 0xF0 and data[index + 1] < 0x90) or (first == 0xF4 and data[index + 1] >= 0x90))
        if valid:
            index += width
        else:
            invalid += 1
            index += 1
    return invalid


def generation_metrics(group: list[dict[str, str]]) -> dict[str, float | int]:
    values = []
    for row in group:
        tokens = token_ids(row["token_ids"])
        data = bytes.fromhex(row["bytes_hex"])
        unique = len(set(tokens)) / len(tokens) if tokens else 0.0
        repeated_tokens = (
            sum(token in tokens[:index] for index, token in enumerate(tokens)) / len(tokens)
            if tokens
            else 0.0
        )
        lines = [line for line in data.splitlines() if line.strip()]
        duplicate_lines = (
            (len(lines) - len(set(lines))) / len(lines) if lines else 0.0
        )
        controls = sum(
            byte < 0x20 and byte not in {0x09, 0x0A} or byte == 0x7F for byte in data
        )
        values.append(
            {
                "repeated_token_fraction": repeated_tokens,
                "repeated_3gram_fraction": repeated_fraction(tokens, 3),
                "repeated_4gram_fraction": repeated_fraction(tokens, 4),
                "longest_repeated_span": longest_repeat(tokens),
                "duplicate_line_rate": duplicate_lines,
                "unique_token_ratio": unique,
                "distinct_2": 1.0 - repeated_fraction(tokens, 2),
                "distinct_3": 1.0 - repeated_fraction(tokens, 3),
                "premature_eod": int(row["eod"]),
                "reached_max": int(row["reached_max"]),
                "invalid_utf8_bytes": invalid_utf8_bytes(data),
                "escaped_control_bytes": controls,
                "generated_tokens": len(tokens),
            }
        )
    return {
        "samples": len(values),
        **{
            key: sum(float(value[key]) for value in values) / len(values)
            for key in values[0]
        },
    }


def analyze_generation(args: argparse.Namespace) -> None:
    generations = rows(args.directory / "generations.csv")
    explorer = rows(args.directory / "next-token-explorer.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in generations:
        grouped[(row["recipe"], row["requested_tokens"])].append(row)
    by_prompt: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in explorer:
        by_prompt[row["prompt_index"]].append(row)
    for prompt, group in by_prompt.items():
        ranks = [int(row["rank"]) for row in group]
        probabilities = [float(row["probability"]) for row in group]
        totals = [float(row["full_probability_sum"]) for row in group]
        if ranks != [1, 2, 3, 4, 5] or probabilities != sorted(probabilities, reverse=True):
            raise ValueError(f"invalid next-token ordering for prompt {prompt}")
        if any(not math.isfinite(value) for value in probabilities + totals) or any(
            abs(value - 1.0) > 1.0e-10 for value in totals
        ):
            raise ValueError(f"invalid next-token probabilities for prompt {prompt}")
    result = {
        "recipes": {
            f"{recipe}_tokens_{length}": generation_metrics(group)
            for (recipe, length), group in sorted(grouped.items())
        },
        "next_token_explorer": {
            "prompts": len(by_prompt),
            "candidates_per_prompt": 5,
            "finite_sorted_and_normalized": True,
            "eod_candidates": sum(int(row["is_eod"]) for row in explorer),
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def freeze_spec(args: argparse.Namespace) -> None:
    specification: dict[str, Any] = {
        "format": "LEDA_AUTOCOMPLETE_EVALUATION_V1",
        "model": {
            "name": "Leda Demo v0",
            "checkpoint_sha256": sha256(args.checkpoint),
            "parameters": 40_385_024,
            "context_capacity": 512,
        },
        "tokenizer": {"format": "SPARTOKN v1", "sha256": sha256(args.tokenizer)},
        "corpus": {"fingerprint": args.corpus_fingerprint},
        "tool_commit": args.tool_commit,
        "default_decoding": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "selection": {
            "quantitative_seed": 3301,
            "next_token_full_chunks_per_domain": 20,
            "next_token_positions_per_full_chunk": 511,
            "prefix_examples_per_domain": 8,
            "prefix_lengths": [32, 64, 128, 256, 400],
            "continuation_targets": 64,
            "bootstrap_seed": 3302,
            "bootstrap_samples": 10_000,
            "generation_seeds": [3303, 3304, 3305],
        },
        "metric_definitions": {
            "lm_ce": "token-weighted negative log likelihood; documents reset context; overlapping 512-token chunks score each target exactly once",
            "perplexity": "exp(token-weighted CE)",
            "confidence_interval": "document bootstrap percentile 95% interval",
            "next_token": "top-k correctness, correct-token softmax probability, and entropy over deterministic full 512-token chunks",
            "prefix": "teacher-forced CE over the same 64 true continuation tokens",
            "context_ablation": "same target continuation with only the final 32/64/128/256/400 prefix tokens available",
        },
        "precision": "Float32 weights with full-precision matmul",
        "test_policy": "one evaluation after this specification is hashed; no tuning from test results",
    }
    payload = (json.dumps(specification, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(json.dumps({"evaluation_spec": str(args.output), "sha256": digest}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("directory", type=pathlib.Path)
    metrics.add_argument("--bootstrap-samples", type=int, default=10_000)
    metrics.add_argument("--bootstrap-seed", type=int, default=3302)
    metrics.add_argument("--output", type=pathlib.Path)
    metrics.set_defaults(function=analyze_metrics)
    generation = commands.add_parser("generation")
    generation.add_argument("directory", type=pathlib.Path)
    generation.add_argument("--output", type=pathlib.Path)
    generation.set_defaults(function=analyze_generation)
    freeze = commands.add_parser("freeze-spec")
    freeze.add_argument("checkpoint", type=pathlib.Path)
    freeze.add_argument("tokenizer", type=pathlib.Path)
    freeze.add_argument("output", type=pathlib.Path)
    freeze.add_argument("--corpus-fingerprint", required=True)
    freeze.add_argument("--tool-commit", required=True)
    freeze.add_argument("--temperature", type=float, required=True)
    freeze.add_argument("--top-k", type=int, required=True)
    freeze.add_argument("--top-p", type=float, required=True)
    freeze.add_argument("--max-new-tokens", type=int, required=True)
    freeze.set_defaults(function=freeze_spec)
    args = parser.parse_args()
    if getattr(args, "bootstrap_samples", 1) <= 0:
        raise ValueError("bootstrap sample count must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
