#!/usr/bin/env python3
"""
plot_seq_sweep.py — Plot total layer time vs sequence position for all CUDA variants.

Reads the per-impl accumulating CSVs produced by run_profiling_experiment.sh, groups rows
by the 'pos' column, and generates:
  1. docs/profiling_plots/seq_sweep_total_layer_ms.png  — all impls on one chart
  2. docs/profiling_plots/seq_sweep_stages_<impl>.png   — per-stage stacked bar per impl

Usage:
    python plot_seq_sweep.py
    python plot_seq_sweep.py --results-root profiling_runs --output-dir docs/profiling_plots
    python plot_seq_sweep.py --impls cublas,cublas_fp16,flash,tiled
    python plot_seq_sweep.py --col total_layer_ms --min-n 1
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for headless/WSL
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── constants ────────────────────────────────────────────────────────────────

ALL_IMPLS = ["cublas", "flash", "tiled", "cublas_fp16", "cublas_bf16", "cublas_fp16tc"]

# Canonical stage columns shared across flash/cublas variants.
# tiled uses cpu_attn_ms / h2d_out_proj_res_ms; those are added if present.
STAGE_COLS_CANONICAL = [
    "rms_att_ms",
    "qkv_ms",
    "rope_kvcache_ms",
    "flash_attn_ms",
    "out_proj_res_ms",
    "rms_ffn_ms",
    "ffn_w1w3_ms",
    "swiglu_w2_ms",
]
STAGE_COLS_TILED_EXTRA = ["cpu_attn_ms", "h2d_out_proj_res_ms"]

IMPL_COLORS = {
    "cublas":        "#1f77b4",
    "flash":         "#ff7f0e",
    "tiled":         "#2ca02c",
    "cublas_fp16":   "#d62728",
    "cublas_bf16":   "#9467bd",
    "cublas_fp16tc": "#8c564b",
}
IMPL_LABELS = {
    "cublas":        "cublas (FP32)",
    "flash":         "flash (custom kernel)",
    "tiled":         "tiled (custom matmul)",
    "cublas_fp16":   "cublas_fp16 (FP16 weights)",
    "cublas_bf16":   "cublas_bf16 (BF16 weights)",
    "cublas_fp16tc": "cublas_fp16tc (FP16 tensor-core)",
}
STAGE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_pos(rows, col="total_layer_ms"):
    """Return dict pos(int) -> list[float], skipping rows that lack pos or col."""
    groups = defaultdict(list)
    for row in rows:
        try:
            groups[int(row["pos"])].append(float(row[col]))
        except (KeyError, ValueError):
            pass
    return groups


def stats(values):
    """Return (mean, stderr) for list[float]."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (var / n) ** 0.5


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot total layer time vs sequence position for CUDA variants."
    )
    p.add_argument("--results-root", default="profiling_runs",
                   help="Root directory containing per-impl CSVs (default: profiling_runs)")
    p.add_argument("--output-dir", default="docs/profiling_plots",
                   help="Directory to write output plots (default: docs/profiling_plots)")
    p.add_argument("--impls", default="",
                   help="Comma-separated list of impls to include (default: all found)")
    p.add_argument("--col", default="total_layer_ms",
                   help="CSV column to plot on the y-axis (default: total_layer_ms)")
    p.add_argument("--min-n", type=int, default=1,
                   help="Minimum measurements at a position to include it (default: 1)")
    args = p.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    impls_filter = [s.strip() for s in args.impls.split(",")] if args.impls else None

    # ── load per-impl data ────────────────────────────────────────────────────
    data = {}        # impl -> {pos: (mean, stderr)}
    raw_rows = {}    # impl -> list[dict]
    found_impls = []

    for impl in ALL_IMPLS:
        if impls_filter and impl not in impls_filter:
            continue
        csv_path = results_root / impl / f"profiling_{impl}.csv"
        if not csv_path.exists():
            continue
        rows = load_csv(csv_path)
        if not rows:
            continue
        raw_rows[impl] = rows
        groups = group_by_pos(rows, col=args.col)
        impl_data = {}
        for pos, vals in sorted(groups.items()):
            if len(vals) < args.min_n:
                continue
            impl_data[pos] = stats(vals)
        if impl_data:
            data[impl] = impl_data
            found_impls.append(impl)

    if not data:
        print(
            f"No data found under '{results_root}'. "
            "Run the profiling sweep first with --profile-pos-list.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found data for: {', '.join(found_impls)}")

    # ── print summary table ───────────────────────────────────────────────────
    all_pos = sorted({pos for impl_data in data.values() for pos in impl_data})
    col_w = 20
    print(f"\nSeq-length sweep — {args.col}:")
    header = f"{'pos':>6}  " + "  ".join(f"{impl:>{col_w}}" for impl in found_impls)
    print(header)
    print("-" * len(header))
    for pos in all_pos:
        parts = []
        for impl in found_impls:
            if pos in data[impl]:
                mean, se = data[impl][pos]
                parts.append(f"{mean:>9.4f}±{se:<9.4f}")
            else:
                parts.append(f"{'—':>{col_w}}")
        print(f"{pos:>6}  " + "  ".join(parts))

    if len(all_pos) < 2:
        print(
            "\nOnly one unique position found — run with --profile-pos-list to collect "
            "multiple positions before generating a sweep chart.",
            file=sys.stderr,
        )
        if not HAS_MATPLOTLIB:
            return
        # Still generate a single-point plot so users can see the framework works.

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib not installed — skipping plot generation.")
        print("Install with:  pip install matplotlib")
        return

    # ── Plot 1: total_layer_ms vs pos, all impls ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for impl in found_impls:
        impl_data = data[impl]
        positions = sorted(impl_data)
        means = [impl_data[pos][0] for pos in positions]
        errs  = [impl_data[pos][1] for pos in positions]
        ax.errorbar(
            positions, means, yerr=errs,
            label=IMPL_LABELS.get(impl, impl),
            color=IMPL_COLORS.get(impl),
            marker="o", linewidth=2, capsize=4, markersize=6,
        )

    ax.set_xlabel("Sequence position (pos)", fontsize=13)
    col_label = args.col.replace("_", " ")
    ax.set_ylabel(f"{col_label} (ms)", fontsize=13)
    ax.set_title(
        "Layer-0 time vs Sequence Position\n(Llama2 7B · Modal A100)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    out1 = output_dir / f"seq_sweep_{args.col}.png"
    fig.tight_layout()
    fig.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")
    plt.close(fig)

    # ── Plot 2: per-stage stacked bar for each impl (only when ≥2 positions) ──
    for impl in found_impls:
        rows = raw_rows.get(impl, [])
        if not rows:
            continue
        # Determine which stage cols are present for this impl
        avail_stages = [c for c in STAGE_COLS_CANONICAL + STAGE_COLS_TILED_EXTRA
                        if c in rows[0]]
        impl_positions = sorted(data[impl])
        if len(impl_positions) < 2 or not avail_stages:
            continue

        # Group by pos for each stage column
        stage_by_pos = defaultdict(lambda: defaultdict(list))
        for row in rows:
            try:
                pos = int(row["pos"])
                for col in avail_stages:
                    stage_by_pos[pos][col].append(float(row[col]))
            except (KeyError, ValueError):
                pass

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bottom = [0.0] * len(impl_positions)

        for idx, col in enumerate(avail_stages):
            vals = [
                sum(stage_by_pos[pos][col]) / len(stage_by_pos[pos][col])
                if stage_by_pos[pos][col] else 0.0
                for pos in impl_positions
            ]
            ax2.bar(
                impl_positions, vals, bottom=bottom,
                label=col.replace("_", " "),
                color=STAGE_PALETTE[idx % len(STAGE_PALETTE)],
                width=max(3, impl_positions[0] * 0.35) if impl_positions else 10,
                edgecolor="white", linewidth=0.5,
            )
            bottom = [b + v for b, v in zip(bottom, vals)]

        ax2.set_xlabel("Sequence position (pos)", fontsize=13)
        ax2.set_ylabel("Time (ms)", fontsize=13)
        ax2.set_title(
            f"Stage breakdown vs pos — {IMPL_LABELS.get(impl, impl)}\n(Llama2 7B · Modal A100)",
            fontsize=13,
        )
        ax2.legend(fontsize=8, loc="upper left", ncol=2)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)

        out2 = output_dir / f"seq_sweep_stages_{impl}.png"
        fig2.tight_layout()
        fig2.savefig(out2, dpi=150)
        print(f"Saved: {out2}")
        plt.close(fig2)


if __name__ == "__main__":
    main()
