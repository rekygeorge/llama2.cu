#!/usr/bin/env python3
"""Compute statistical profiling metrics from a profiling CSV.

Outputs per-metric summary with:
- n, mean, std, min, max, median, p95, coefficient of variation
- 95% confidence interval of the mean (normal approximation)

Usage:
    python profile_stats.py --input profiling_flash_run_01.csv
    python profile_stats.py --input profiling_flash_run_01.csv --output profile_stats_summary.csv
"""

import argparse
import csv
import math
from statistics import mean, median, stdev

PROFILE_COLUMNS = [
    "rms_att_ms",
    "qkv_ms",
    "rope_kvcache_ms",
    "flash_attn_ms",
    "out_proj_res_ms",
    "rms_ffn_ms",
    "ffn_w1w3_ms",
    "swiglu_w2_ms",
    "total_layer_ms",
    "est_32_layer_ms",
    "est_tok_s",
]

NON_METRIC_COLUMNS = {
    "position",
    "pos",
    "token_pos",
    "step",
    "iteration",
    "layer",
}


def percentile(values, p):
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_vals[low]
    w = rank - low
    return sorted_vals[low] * (1.0 - w) + sorted_vals[high] * w


def summarize(values):
    n = len(values)
    mu = mean(values)
    sigma = stdev(values) if n > 1 else 0.0
    vmin = min(values)
    vmax = max(values)
    med = median(values)
    p95 = percentile(values, 0.95)
    cv_pct = (sigma / mu * 100.0) if mu != 0.0 else float("nan")
    ci95 = 1.96 * sigma / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mu,
        "std": sigma,
        "min": vmin,
        "max": vmax,
        "median": med,
        "p95": p95,
        "cv_pct": cv_pct,
        "ci95_low": mu - ci95,
        "ci95_high": mu + ci95,
    }


def _can_parse_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def detect_metric_columns(rows):
    if not rows:
        raise ValueError("No rows found in input CSV")

    first = rows[0]
    available = list(first.keys())

    # Backward-compatible path for flash/cublas schema.
    if all(col in first for col in PROFILE_COLUMNS):
        return PROFILE_COLUMNS

    numeric_cols = []
    for col in available:
        if col.lower() in NON_METRIC_COLUMNS:
            continue
        if any(_can_parse_float(r.get(col, "")) for r in rows):
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError(
            "Could not detect any numeric metric columns. "
            "CSV headers found: " + ", ".join(available)
        )

    return numeric_cols


def load_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("No rows found in input CSV")
    metric_cols = detect_metric_columns(rows)
    return rows, metric_cols


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for CUDA profiling CSV")
    parser.add_argument("--input", required=True, help="Input profiling CSV path")
    parser.add_argument("--output", default="", help="Optional summary CSV output path")
    parser.add_argument(
        "--warmup-skip",
        type=int,
        default=0,
        metavar="K",
        help="Drop the first K rows before computing statistics (default: 0). "
             "Use 3-5 to discard cold-start GPU runs.",
    )
    args = parser.parse_args()

    rows, metric_cols = load_rows(args.input)

    total_rows = len(rows)
    if args.warmup_skip > 0:
        if args.warmup_skip >= total_rows:
            raise ValueError(
                f"--warmup-skip {args.warmup_skip} >= total rows {total_rows}; "
                "nothing left to summarise."
            )
        rows = rows[args.warmup_skip:]

    summaries = {}
    for col in metric_cols:
        vals = [float(r[col]) for r in rows]
        summaries[col] = summarize(vals)

    print("=== Profiling Statistical Summary ===")
    print(f"Input rows: {total_rows} (skipped first {args.warmup_skip} warmup row(s), analysed {len(rows)})")
    if metric_cols != PROFILE_COLUMNS:
        print("Detected non-default CSV schema; summarizing numeric metric columns:")
        print(", ".join(metric_cols))

    for col in metric_cols:
        s = summaries[col]
        print(f"\n[{col}]")
        print(f"  n={s['n']}")
        print(f"  mean={s['mean']:.6f}")
        print(f"  std={s['std']:.6f}")
        print(f"  min={s['min']:.6f}")
        print(f"  max={s['max']:.6f}")
        print(f"  median={s['median']:.6f}")
        print(f"  p95={s['p95']:.6f}")
        print(f"  cv%={s['cv_pct']:.3f}")
        print(f"  ci95=[{s['ci95_low']:.6f}, {s['ci95_high']:.6f}]")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "metric",
                "n",
                "warmup_skipped",
                "mean",
                "std",
                "min",
                "max",
                "median",
                "p95",
                "cv_pct",
                "ci95_low",
                "ci95_high",
            ])
            for col in metric_cols:
                s = summaries[col]
                writer.writerow([
                    col,
                    s["n"],
                    args.warmup_skip,
                    f"{s['mean']:.8f}",
                    f"{s['std']:.8f}",
                    f"{s['min']:.8f}",
                    f"{s['max']:.8f}",
                    f"{s['median']:.8f}",
                    f"{s['p95']:.8f}",
                    f"{s['cv_pct']:.8f}",
                    f"{s['ci95_low']:.8f}",
                    f"{s['ci95_high']:.8f}",
                ])
        print(f"\nWrote summary CSV to: {args.output}")


if __name__ == "__main__":
    main()
