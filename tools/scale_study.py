#!/usr/bin/env python3
"""Analyze Leda scaling-run metrics and the fixed Phase-28 tokenizer sample."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import statistics


CANDIDATES = {
    "A": (128, 384, 4, 4, 2),
    "B": (256, 768, 8, 8, 2),
    "C": (384, 1152, 12, 6, 2),
    "D": (512, 1536, 12, 8, 2),
}


def parameter_count(vocab: int, model: tuple[int, int, int, int, int]) -> int:
    dimension, hidden, layers, query_heads, kv_heads = model
    head_dimension = dimension // query_heads
    per_layer = (
        2 * dimension * dimension
        + 2 * dimension * kv_heads * head_dimension
        + 2 * head_dimension
        + 2 * dimension
        + 3 * dimension * hidden
    )
    return vocab * dimension + layers * per_layer + dimension


def make_spec(args: argparse.Namespace) -> None:
    dimension, hidden, layers, query_heads, kv_heads = CANDIDATES[args.model]
    parameters = parameter_count(args.model_vocab_size, CANDIDATES[args.model])
    intended_tokens = (
        args.maximum_steps
        * args.microbatch_size
        * args.accumulation_steps
        * (args.sequence_length - 1)
    )
    fields = [
        ("format", "LEDA_SCALE_RUN_V1"),
        ("spar_commit", args.spar_commit),
        ("leda_commit", args.leda_commit),
        ("train_manifest", args.train_manifest),
        ("train_manifest_sha256", args.train_manifest_sha256),
        ("validation_manifest", args.validation_manifest),
        ("validation_manifest_sha256", args.validation_manifest_sha256),
        ("tokenizer_vocab_size", args.tokenizer_vocab_size),
        ("model_vocab_size", args.model_vocab_size),
        ("seed", args.seed),
        ("model_dim", dimension),
        ("hidden_dim", hidden),
        ("num_layers", layers),
        ("num_query_heads", query_heads),
        ("num_kv_heads", kv_heads),
        ("parameters", parameters),
        ("sequence_length", args.sequence_length),
        ("stride", args.stride),
        ("microbatch_size", args.microbatch_size),
        ("accumulation_steps", args.accumulation_steps),
        ("peak_learning_rate", args.peak_learning_rate),
        ("minimum_learning_rate", args.minimum_learning_rate),
        ("warmup_steps", args.warmup_steps),
        ("schedule_steps", args.schedule_steps),
        ("max_grad_norm", args.max_grad_norm),
        ("beta1", args.beta1),
        ("beta2", args.beta2),
        ("epsilon", args.epsilon),
        ("weight_decay", args.weight_decay),
        ("shuffle_seed", args.shuffle_seed),
        ("precision", args.precision),
        ("maximum_steps", args.maximum_steps),
        ("intended_tokens", intended_tokens),
        ("validation_steps", args.validation_steps),
        ("checkpoint_steps", args.checkpoint_steps),
        ("save_final", int(args.save_final)),
    ]
    contents = "".join(f"{key}={value}\n" for key, value in fields)
    args.output.write_text(contents, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parameters": parameters, "intended_tokens": intended_tokens}))


def tokenizer_bias(args: argparse.Namespace) -> None:
    records: dict[str, dict] = {}
    with args.provenance.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["split"] == "train":
                records[str(pathlib.Path(record["document_path"]).resolve())] = record
    paths = [pathlib.Path(line).resolve() for line in args.train_paths.read_text().splitlines() if line]
    sample: list[dict] = []
    sample_bytes = 0
    for path in paths:
        record = records[str(path)]
        size = int(record["content_bytes"])
        if sample and (sample_bytes >= args.maximum_bytes or size > args.maximum_bytes - sample_bytes):
            break
        sample.append(record)
        sample_bytes += size

    def categories(items: list[dict]) -> collections.Counter[str]:
        result: collections.Counter[str] = collections.Counter()
        for item in items:
            result.update(item["science_domain_category"])
        return result

    full = [records[str(path)] for path in paths]
    sample_categories = categories(sample)
    full_categories = categories(full)
    names = set(sample_categories) | set(full_categories)
    differences = []
    for name in names:
        sample_share = sample_categories[name] / len(sample)
        full_share = full_categories[name] / len(full)
        differences.append(
            {
                "category": name,
                "sample_document_share": sample_share,
                "full_document_share": full_share,
                "difference": sample_share - full_share,
            }
        )
    differences.sort(key=lambda item: abs(item["difference"]), reverse=True)
    pmcids = [int(item["stable_source_id"].removeprefix("PMC")) for item in sample]
    print(
        json.dumps(
            {
                "sample_documents": len(sample),
                "sample_bytes": sample_bytes,
                "full_train_documents": len(full),
                "sample_fraction_documents": len(sample) / len(full),
                "path_order_is_sorted": paths == sorted(paths),
                "sample_pmcid_min": min(pmcids),
                "sample_pmcid_max": max(pmcids),
                "retrieval_dates": sorted({item["retrieval_date"] for item in sample}),
                "largest_category_share_differences": differences[:20],
            },
            indent=2,
            sort_keys=True,
        )
    )


def finite_float(text: str) -> float | None:
    value = float(text)
    return value if math.isfinite(value) else None


def run_summary(directory: pathlib.Path, warmup: int) -> dict:
    with (directory / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    trained = [row for row in rows if int(row["step"]) > 0]
    timed = [row for row in trained if int(row["step"]) > warmup]
    update_ms = [float(row["update_ms"]) for row in timed]
    gradients = [float(row["gradient_norm"]) for row in trained]
    validations = [
        {"step": int(row["step"]), "tokens": int(row["tokens_seen"]), "loss": float(row["validation_mean_loss"])}
        for row in rows
        if finite_float(row["validation_mean_loss"]) is not None
    ]
    validation_uncertainty = []
    for item in validations:
        path = directory / f"validation-{item['step']}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            losses = [float(row["mean_loss"]) for row in csv.DictReader(stream)]
        mean = statistics.fmean(losses)
        stddev = statistics.stdev(losses) if len(losses) > 1 else 0.0
        standard_error = stddev / math.sqrt(len(losses))
        validation_uncertainty.append(
            {
                **item,
                "windows": len(losses),
                "window_mean": mean,
                "window_stddev": stddev,
                "standard_error": standard_error,
                "normal_95_low": mean - 1.96 * standard_error,
                "normal_95_high": mean + 1.96 * standard_error,
            }
        )
    return {
        "run_directory": str(directory),
        "updates": len(trained),
        "final_step": int(trained[-1]["step"]) if trained else 0,
        "final_tokens": int(trained[-1]["tokens_seen"]) if trained else 0,
        "median_update_ms_after_warmup": statistics.median(update_ms) if update_ms else None,
        "median_tokens_per_second_after_warmup": (
            statistics.median(float(row["tokens_per_second"]) for row in timed) if timed else None
        ),
        "gpu_update_seconds": sum(float(row["update_ms"]) for row in trained) / 1000.0,
        "validation_seconds": sum(float(row["validation_ms"]) for row in rows) / 1000.0,
        "clipped_updates": sum(int(row["clipped"]) for row in trained),
        "clipping_frequency": (
            sum(int(row["clipped"]) for row in trained) / len(trained) if trained else None
        ),
        "gradient_norm_mean": statistics.fmean(gradients) if gradients else None,
        "gradient_norm_stddev": statistics.stdev(gradients) if len(gradients) > 1 else 0.0,
        "gradient_norm_min": min(gradients) if gradients else None,
        "gradient_norm_max": max(gradients) if gradients else None,
        "nonfinite_metric_rows": sum(
            any(finite_float(row[key]) is None for key in ("train_mean_loss", "gradient_norm", "clip_scale"))
            for row in trained
        ),
        "validations": validations,
        "validation_uncertainty": validation_uncertainty,
    }


def summarize(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            [run_summary(directory.resolve(), args.warmup) for directory in args.run_directories],
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    bias = commands.add_parser("tokenizer-bias")
    bias.add_argument("provenance", type=pathlib.Path)
    bias.add_argument("train_paths", type=pathlib.Path)
    bias.add_argument("maximum_bytes", type=int)
    bias.set_defaults(function=tokenizer_bias)
    runs = commands.add_parser("summarize")
    runs.add_argument("run_directories", type=pathlib.Path, nargs="+")
    runs.add_argument("--warmup", type=int, default=5)
    runs.set_defaults(function=summarize)
    spec = commands.add_parser("make-spec")
    spec.add_argument("output", type=pathlib.Path)
    spec.add_argument("--model", choices=sorted(CANDIDATES), required=True)
    spec.add_argument("--spar-commit", required=True)
    spec.add_argument("--leda-commit", required=True)
    spec.add_argument("--train-manifest", required=True)
    spec.add_argument("--train-manifest-sha256", required=True)
    spec.add_argument("--validation-manifest", required=True)
    spec.add_argument("--validation-manifest-sha256", required=True)
    spec.add_argument("--tokenizer-vocab-size", type=int, default=8192)
    spec.add_argument("--model-vocab-size", type=int, default=8193)
    spec.add_argument("--seed", type=int, default=2901)
    spec.add_argument("--sequence-length", type=int, default=128)
    spec.add_argument("--stride", type=int, default=128)
    spec.add_argument("--microbatch-size", type=int, required=True)
    spec.add_argument("--accumulation-steps", type=int, required=True)
    spec.add_argument("--peak-learning-rate", type=float, required=True)
    spec.add_argument("--minimum-learning-rate", type=float, required=True)
    spec.add_argument("--warmup-steps", type=int, required=True)
    spec.add_argument("--schedule-steps", type=int, required=True)
    spec.add_argument("--max-grad-norm", type=float, default=1.0)
    spec.add_argument("--beta1", type=float, default=0.9)
    spec.add_argument("--beta2", type=float, default=0.95)
    spec.add_argument("--epsilon", type=float, default=1e-8)
    spec.add_argument("--weight-decay", type=float, default=0.1)
    spec.add_argument("--shuffle-seed", type=int, default=2902)
    spec.add_argument("--precision", choices=("full", "fp16"), required=True)
    spec.add_argument("--maximum-steps", type=int, required=True)
    spec.add_argument("--validation-steps", default="none")
    spec.add_argument("--checkpoint-steps", default="none")
    spec.add_argument("--save-final", action="store_true")
    spec.set_defaults(function=make_spec)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
