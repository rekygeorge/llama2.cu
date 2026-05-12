#!/usr/bin/env python3
"""
roofline.py — Roofline analysis for Llama2 7B CUDA variants on A100.

Reads per-impl profile_stats CSV files (produced by profile_stats.py), computes
the arithmetic intensity and achieved GFLOPs/s for every measured stage, then:

  1. Prints a text table (Step 4c) with attainment % vs memory-bandwidth ceiling
     and compute ceiling.
  2. Saves docs/profiling_plots/roofline.png — classic roofline scatter plot with
     sloped memory-bandwidth ceiling, two horizontal compute ceilings (FP32 and
     FP16 Tensor Core), and one scatter point per impl × stage combination.

Usage:
    python roofline.py
    python roofline.py --results-root profiling_runs --output-dir docs/profiling_plots
    python roofline.py --no-plot   # text table only

A100 hardware constants used (SXM4-40GB) — source: NVIDIA A100 Whitepaper:
    Peak FP32 (CUDA cores)        :  19.5 TFLOP/s
    Peak TF32 TC (dense)          :  77.6 TFLOP/s  (structured inputs; not used at batch=1 GEMV)
    Peak FP16/BF16 TC (dense)     : 312   TFLOP/s  (CUBLAS_COMPUTE_16F / COMPUTE_BF16)
    Peak FP16 TC (sparse 2:4)     : 624   TFLOP/s  (requires explicit structured sparsity)
    Peak memory bandwidth         : 2000  GB/s     (HBM2e)

Ridge points  (I* = peak_compute / peak_BW):
    FP32 CUDA cores  : 19.5e12 / 2000e9 =   9.75 FLOP/byte
    TF32 TC (dense)  : 77.6e12 / 2000e9 =  38.8  FLOP/byte
    FP16/BF16 TC (dense): 312e12 / 2000e9 = 156   FLOP/byte

Note: at batch=1 (autoregressive decode), all weight projections are GEMVs.
cuBLAS does NOT engage tensor cores for GEMV regardless of COMPUTE_16F flag.
The effective compute ceiling is always 19.5 TFLOP/s (FP32 CUDA cores) at batch=1.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── A100 hardware constants ───────────────────────────────────────────────────

GPU_NAME            = "NVIDIA A100-SXM4-40GB"
PEAK_BW_TBs         = 2.0            # TB/s  (2000 GB/s HBM2e)
PEAK_BW_GBs         = PEAK_BW_TBs * 1000   # GB/s
PEAK_FP32_TFLOPS    = 19.5           # TFLOP/s  — FP32 CUDA cores (non-tensor-core)
PEAK_TF32TC_TFLOPS  = 77.6           # TFLOP/s  — TF32 Tensor Core, dense (Ampere)
PEAK_FP16TC_TFLOPS  = 312.0          # TFLOP/s  — FP16/BF16 Tensor Core, dense (Ampere)
# Note: 624 TFLOP/s FP16-TC (sparse) requires explicit 2:4 structured sparsity;
#       cuBLAS does not use it automatically.

# Ridgeline points: I* = peak_compute / peak_BW  (FLOP/byte)
# These are the arithmetic intensities at which the kernel transitions from
# memory-bound to compute-bound.
RIDGE_FP32    = (PEAK_FP32_TFLOPS   * 1e12) / (PEAK_BW_GBs * 1e9)   #  9.75 FLOP/byte
RIDGE_TF32TC  = (PEAK_TF32TC_TFLOPS * 1e12) / (PEAK_BW_GBs * 1e9)   # 38.8  FLOP/byte
RIDGE_FP16TC  = (PEAK_FP16TC_TFLOPS * 1e12) / (PEAK_BW_GBs * 1e9)   # 156   FLOP/byte

# ── Llama2-7B model dimensions ────────────────────────────────────────────────

DIM         = 4096    # transformer (hidden) dimension
HIDDEN_DIM  = 11008   # FFN intermediate dimension
N_HEADS     = 32
N_KV_HEADS  = 32
HEAD_SIZE   = DIM // N_HEADS   # 128

BYTES_FP32  = 4
BYTES_FP16  = 2
BYTES_BF16  = 2

# ── Per-stage FLOP and byte-traffic formulae ──────────────────────────────────
#
# Convention:
#   FLOPs = useful multiply-add operations × 2  (one multiply + one add = 2 FLOP)
#   Bytes = weight bytes read + activation bytes read + output bytes written
#
# All values are for a single token (batch=1, seq_len matters only for attention).
# We use pos=50 as the representative position for attention stages since that is
# the profiling trigger position used in our experiments.
#
# Legend for byte formulas:
#   w_bytes = per-weight-element dtype size (4 for FP32, 2 for FP16/BF16)
#   act_bytes = 4 (activations stay FP32 in all variants)

POS = 50   # representative sequence position used in our experiments


def stage_specs(w_bytes: int, pos: int = POS):
    """
    Return a dict of stage_name -> (flops, bytes_traffic) for one forward token.

    w_bytes  : bytes per weight element (4=FP32, 2=FP16/BF16)
    pos      : current KV-cache length (number of past tokens including current)
    """
    act = BYTES_FP32   # activations are always FP32

    # ── RMSNorm (attention) ──────────────────────────────────────────────────
    # Reads: input dim floats + weight dim floats; Writes: output dim floats
    # FLOPs: sum-of-squares (dim MACs) + reciprocal sqrt + dim multiplies ≈ 3·dim
    rms_flops = 3 * DIM
    rms_bytes = act * DIM + w_bytes * DIM + act * DIM
    # (weight bytes: norm weights are stored as FP32 in all our variants)
    rms_bytes = act * DIM + act * DIM + act * DIM   # weight always FP32 for norms

    # ── QKV Projections (3 matrix-vector products) ───────────────────────────
    # Q: (dim,dim), K: (kv_dim, dim), V: (kv_dim, dim); kv_dim = n_kv_heads * head_size
    kv_dim = N_KV_HEADS * HEAD_SIZE  # = dim for Llama2-7B (no GQA, 32 kv heads)
    q_flops  = 2 * DIM * DIM
    kv_flops = 2 * kv_dim * DIM * 2     # K and V
    qkv_flops = q_flops + kv_flops
    # Weight matrices: Q is (dim, dim), K+V are (kv_dim, dim) each
    qkv_w_bytes = w_bytes * (DIM * DIM + 2 * kv_dim * DIM)
    qkv_act_in  = act * DIM                           # input vector x
    qkv_act_out = act * (DIM + 2 * kv_dim)            # q, k, v output vectors
    qkv_bytes   = qkv_w_bytes + qkv_act_in + qkv_act_out

    # ── RoPE + KV Cache ──────────────────────────────────────────────────────
    # RoPE: 2 reads + 2 writes per pair, DIM/2 + kv_dim/2 pairs
    rope_flops = 6 * (DIM + kv_dim)   # sin/cos multiply-adds per pair
    rope_bytes = act * (DIM + kv_dim) * 2  # read q,k + write q,k

    # ── Flash Attention ──────────────────────────────────────────────────────
    # For each head: QK^T dot products = pos * head_size multiply-adds
    # Softmax → weighted sum with V = pos * head_size multiply-adds
    # Total per head: 4 * pos * head_size FLOPs (QK, softmax, PV, output)
    # All attention computations are in FP32.
    attn_flops = N_HEADS * (4 * pos * HEAD_SIZE)
    # Read: Q (dim), K cache (pos * kv_dim), V cache (pos * kv_dim)
    # Write: output (dim)  — all FP32 for attention
    attn_bytes = (
        act * DIM +                        # Q
        act * pos * kv_dim +               # K cache
        act * pos * kv_dim +               # V cache
        act * DIM                          # output
    )

    # ── Output Projection + Residual ─────────────────────────────────────────
    # W_o: (dim, dim) matrix-vector product
    op_flops = 2 * DIM * DIM
    op_w_bytes = w_bytes * DIM * DIM
    op_act_in  = act * DIM
    op_act_out = act * DIM    # residual add is negligible bytes
    op_bytes   = op_w_bytes + op_act_in + op_act_out

    # ── RMSNorm (FFN) ────────────────────────────────────────────────────────
    rms_ffn_flops = 3 * DIM
    rms_ffn_bytes = act * DIM * 3   # input + weight + output, all FP32

    # ── FFN W1 + W3 ──────────────────────────────────────────────────────────
    # Two (hidden_dim, dim) matrix-vector products
    # Llama2-7B: dim=4096, hidden_dim=11008
    w1w3_flops = 2 * 2 * HIDDEN_DIM * DIM
    w1w3_w_bytes = w_bytes * 2 * HIDDEN_DIM * DIM
    w1w3_act_in  = act * DIM
    w1w3_act_out = act * 2 * HIDDEN_DIM
    w1w3_bytes   = w1w3_w_bytes + w1w3_act_in + w1w3_act_out

    # ── SwiGLU + W2 ──────────────────────────────────────────────────────────
    # SwiGLU: ~3 ops per element (sigmoid, multiply, multiply) on hidden_dim elements
    # W2: (dim, hidden_dim) matrix-vector product
    swiglu_flops = 3 * HIDDEN_DIM + 2 * DIM * HIDDEN_DIM
    swiglu_w_bytes = w_bytes * DIM * HIDDEN_DIM
    swiglu_act_in  = act * 2 * HIDDEN_DIM   # hb + hb2
    swiglu_act_out = act * DIM
    swiglu_bytes   = swiglu_w_bytes + swiglu_act_in + swiglu_act_out

    return {
        "rms_att":    (rms_flops,   rms_bytes),
        "qkv":        (qkv_flops,   qkv_bytes),
        "rope_kv":    (rope_flops,  rope_bytes),
        "flash_attn": (attn_flops,  attn_bytes),
        "out_proj":   (op_flops,    op_bytes),
        "rms_ffn":    (rms_ffn_flops, rms_ffn_bytes),
        "ffn_w1w3":   (w1w3_flops,  w1w3_bytes),
        "swiglu_w2":  (swiglu_flops, swiglu_bytes),
    }


# Map CSV column names → stage_spec keys
CSV_COL_TO_STAGE = {
    "rms_att_ms":      "rms_att",
    "qkv_ms":          "qkv",
    "rope_kvcache_ms": "rope_kv",
    "flash_attn_ms":   "flash_attn",
    "out_proj_res_ms": "out_proj",
    "rms_ffn_ms":      "rms_ffn",
    "ffn_w1w3_ms":     "ffn_w1w3",
    "swiglu_w2_ms":    "swiglu_w2",
}

# Tiled variant uses different column names for two stages
CSV_COL_TO_STAGE_TILED = {
    "cpu_attn_ms":          "flash_attn",
    "h2d_out_proj_res_ms":  "out_proj",
}

# Which dtype (bytes/weight) does each impl use?
IMPL_W_BYTES = {
    "cublas":        BYTES_FP32,
    "flash":         BYTES_FP32,
    "tiled":         BYTES_FP32,
    "cublas_fp16":   BYTES_FP16,
    "cublas_bf16":   BYTES_BF16,
    "cublas_fp16tc": BYTES_FP16,
}

# Which compute ceiling applies for each implementation?
#
# At batch=1 (GEMV), cuBLAS does NOT engage tensor cores regardless of
# COMPUTE_16F flag — tiles are never filled. The effective ceiling for all
# GEMV workloads is the FP32 CUDA-core peak (19.5 TFLOP/s).
#
# For the roofline we assign the *configured* compute path ceiling so the
# plot shows how far below the relevant theoretical limit the kernel sits.
# The batch=1 GEMV constraint is documented separately in the attainment notes.
IMPL_PEAK_TFLOPS = {
    "cublas":        PEAK_FP32_TFLOPS,    # FP32 CUDA cores — 19.5 TFLOP/s
    "flash":         PEAK_FP32_TFLOPS,    # FP32 CUDA cores — 19.5 TFLOP/s
    "tiled":         PEAK_FP32_TFLOPS,    # FP32 CUDA cores — 19.5 TFLOP/s
    "cublas_fp16":   PEAK_FP32_TFLOPS,    # COMPUTE_32F → FP32 CUDA cores (GEMV)
    "cublas_bf16":   PEAK_FP32_TFLOPS,    # COMPUTE_32F → FP32 CUDA cores (GEMV)
    "cublas_fp16tc": PEAK_FP16TC_TFLOPS,  # COMPUTE_16F → FP16-TC dense ceiling (312 TFLOP/s)
                                          # (TC tiles not filled at batch=1, but this is the
                                          #  theoretical limit for the configured compute path)
}

ALL_IMPLS = ["cublas", "flash", "tiled", "cublas_fp16", "cublas_bf16", "cublas_fp16tc"]

IMPL_COLORS = {
    "cublas":        "#1f77b4",
    "flash":         "#ff7f0e",
    "tiled":         "#2ca02c",
    "cublas_fp16":   "#d62728",
    "cublas_bf16":   "#9467bd",
    "cublas_fp16tc": "#8c564b",
}

STAGE_MARKERS = {
    "rms_att":    "o",
    "qkv":        "s",
    "rope_kv":    "^",
    "flash_attn": "D",
    "out_proj":   "P",
    "rms_ffn":    "v",
    "ffn_w1w3":   "*",
    "swiglu_w2":  "X",
}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_stats_csv(path: Path):
    """Return dict metric -> mean_ms from a profile_stats CSV."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                result[row["metric"]] = float(row["mean"])
            except (KeyError, ValueError):
                pass
    return result


# ── Compute roofline points for one impl ──────────────────────────────────────

def compute_points(impl: str, stats: dict, pos: int = POS):
    """
    Returns list of dicts:
        stage, flops, bytes_traffic, intensity, achieved_gflops, ...
    """
    w_bytes = IMPL_W_BYTES.get(impl, BYTES_FP32)
    specs = stage_specs(w_bytes, pos=pos)
    col_map = dict(CSV_COL_TO_STAGE)
    if impl == "tiled":
        col_map.update(CSV_COL_TO_STAGE_TILED)

    points = []
    for csv_col, stage in col_map.items():
        ms_key = csv_col
        if ms_key not in stats:
            continue
        mean_ms = stats[ms_key]
        if mean_ms <= 0:
            continue
        flops, byte_traffic = specs.get(stage, (None, None))
        if flops is None:
            continue
        time_s = mean_ms * 1e-3
        intensity = flops / byte_traffic           # FLOPs/byte
        achieved_gflops = (flops / time_s) / 1e9  # GFLOPs/s

        peak_gflops = IMPL_PEAK_TFLOPS[impl] * 1e3  # TFLOP/s → GFLOPs/s
        bw_ceiling_gflops = intensity * PEAK_BW_GBs  # GFLOPs/s at this intensity

        # Attainment vs memory bandwidth ceiling (since all stages are well below ridge)
        bw_attainment_pct  = 100.0 * achieved_gflops / bw_ceiling_gflops
        cmp_attainment_pct = 100.0 * achieved_gflops / peak_gflops

        points.append({
            "impl":              impl,
            "stage":             stage,
            "csv_col":           csv_col,
            "mean_ms":           mean_ms,
            "flops":             flops,
            "byte_traffic":      byte_traffic,
            "intensity":         intensity,
            "achieved_gflops":   achieved_gflops,
            "bw_ceiling_gflops": bw_ceiling_gflops,
            "peak_gflops":       peak_gflops,
            "bw_attainment_pct": bw_attainment_pct,
            "cmp_attainment_pct": cmp_attainment_pct,
        })
    return points


# ── Pretty table ──────────────────────────────────────────────────────────────

HEADER_FMT = (
    f"{'impl':<14} {'stage':<12} {'time_ms':>8} {'intensity':>12} "
    f"{'achieved':>12} {'bw_ceil':>12} {'bw_att%':>9} {'cmp_att%':>9}"
)

def print_table(all_points):
    print("\n" + "=" * len(HEADER_FMT))
    print(f"Roofline Analysis — {GPU_NAME}")
    print(f"      BW ceiling  = {PEAK_BW_GBs:.0f} GB/s (HBM2e)")
    print(f"      FP32 (CUDA cores) = {PEAK_FP32_TFLOPS:.1f} TFLOP/s  "
          f"| TF32 TC (dense) = {PEAK_TF32TC_TFLOPS:.1f} TFLOP/s  "
          f"| FP16 TC (dense) = {PEAK_FP16TC_TFLOPS:.0f} TFLOP/s")
    print(f"      Ridge: FP32 = {RIDGE_FP32:.2f} FLOP/byte  "
          f"| TF32-TC = {RIDGE_TF32TC:.1f} FLOP/byte  "
          f"| FP16-TC = {RIDGE_FP16TC:.0f} FLOP/byte")
    print(f"      All measured AIs (0.25–1.0 FLOP/byte) are below even the "
          f"FP32 ridge ({RIDGE_FP32:.2f}) → all stages memory-bound.")
    print("=" * len(HEADER_FMT))
    print(HEADER_FMT)
    print("-" * len(HEADER_FMT))

    cur_impl = None
    for pt in all_points:
        if pt["impl"] != cur_impl:
            if cur_impl is not None:
                print()
            cur_impl = pt["impl"]
        print(
            f"{pt['impl']:<14} {pt['stage']:<12} {pt['mean_ms']:>8.4f} "
            f"{pt['intensity']:>12.3f} {pt['achieved_gflops']:>11.2f}G "
            f"{pt['bw_ceiling_gflops']:>11.2f}G "
            f"{pt['bw_attainment_pct']:>8.1f}% "
            f"{pt['cmp_attainment_pct']:>8.3f}%"
        )

    print("-" * len(HEADER_FMT))
    # Key comparison: cublas vs cublas_fp16 QKV
    _highlight_comparison(all_points)


def _highlight_comparison(all_points):
    """Print the key speedup explanation for QKV."""
    qkv = {pt["impl"]: pt for pt in all_points if pt["stage"] == "qkv"}
    if "cublas" in qkv and "cublas_fp16" in qkv:
        fp32 = qkv["cublas"]
        fp16 = qkv["cublas_fp16"]
        speedup = fp32["mean_ms"] / fp16["mean_ms"] if fp16["mean_ms"] > 0 else 0
        intensity_ratio = fp16["intensity"] / fp32["intensity"] if fp32["intensity"] > 0 else 0
        print(
            f"\nKey insight (QKV projection):\n"
            f"  cublas   FP32: intensity={fp32['intensity']:.3f} FLOPs/byte, "
            f"achieved={fp32['achieved_gflops']:.2f} GFLOPs/s, "
            f"BW attainment={fp32['bw_attainment_pct']:.1f}%\n"
            f"  cublas_fp16: intensity={fp16['intensity']:.3f} FLOPs/byte, "
            f"achieved={fp16['achieved_gflops']:.2f} GFLOPs/s, "
            f"BW attainment={fp16['bw_attainment_pct']:.1f}%\n"
            f"  → FP16 doubles arithmetic intensity ({intensity_ratio:.2f}×), "
            f"measured speedup = {speedup:.2f}×\n"
            f"  → Both are far below the ridge point ({RIDGE_FP32:.0f} FLOPs/byte) "
            f"→ memory-bound regime confirmed."
        )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_roofline(all_points, output_path: Path):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plot. Install with: pip install matplotlib")
        return

    import math as _math

    # ── Layout: two side-by-side panels ──────────────────────────────────────
    # Left  panel: full roofline view with all ceilings
    # Right panel: zoom into the measured data (AI 0.1 – 2.0, GFLOPs 0.1 – 2000)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(20, 9))
    fig.subplots_adjust(wspace=0.35)

    # ── Jitter to separate overlapping points ─────────────────────────────────
    # Points cluster at AI ≈ 0.25, 0.49, 0.50, 0.75, 0.999.
    # We apply a small multiplicative jitter per (impl_index, stage) so that
    # within each cluster every marker stays visually distinct.
    IMPL_ORDER = [i for i in ALL_IMPLS if i in {pt["impl"] for pt in all_points}]
    N_IMPL = len(IMPL_ORDER)

    def jitter_x(base_intensity, impl_idx, n_impl):
        """Spread n_impl points symmetrically around base_intensity (log scale)."""
        if n_impl <= 1:
            return base_intensity
        # log-space spread: ±12 % total width, divided evenly
        spread = 0.12
        offset = spread * (impl_idx / (n_impl - 1) - 0.5)
        return base_intensity * (10 ** offset)

    def jitter_y(base_gflops, stage_idx, n_stages_at_cluster):
        """Spread points vertically within a cluster (log scale)."""
        if n_stages_at_cluster <= 1:
            return base_gflops
        spread = 0.15
        offset = spread * (stage_idx / (n_stages_at_cluster - 1) - 0.5)
        return base_gflops * (10 ** offset)

    # Pre-compute jittered positions: group by rounded AI cluster
    AI_ROUND = 2   # decimal places for cluster key
    # Build cluster membership list
    clusters = {}   # cluster_key -> list of (impl, stage, base_x, base_y)
    for pt in all_points:
        key = round(pt["intensity"], AI_ROUND)
        clusters.setdefault(key, []).append(pt)

    jittered = {}   # (impl, stage) -> (jx, jy)
    for key, pts in clusters.items():
        for stage_rank, pt in enumerate(sorted(pts, key=lambda p: list(STAGE_MARKERS).index(p["stage"])
                                              if p["stage"] in STAGE_MARKERS else 99)):
            impl_idx = IMPL_ORDER.index(pt["impl"]) if pt["impl"] in IMPL_ORDER else 0
            jx = jitter_x(pt["intensity"], impl_idx, N_IMPL)
            jy = jitter_y(pt["achieved_gflops"], stage_rank, len(pts))
            jittered[(pt["impl"], pt["stage"])] = (jx, jy)

    # ── Helper: draw ceilings, scatter, and annotations on an axis ───────────
    def draw_panel(ax, x_min, x_max, y_min, y_max, zoom=False):
        xs = [x_min * (x_max / x_min) ** (i / 1000) for i in range(1001)]

        # BW slope
        ax.loglog(xs, [x * PEAK_BW_GBs for x in xs],
                  color="steelblue", linewidth=2.5, linestyle="--",
                  label=f"BW ceiling ({PEAK_BW_GBs:.0f} GB/s)",
                  zorder=1)

        # Compute ceilings
        ax.axhline(PEAK_FP32_TFLOPS * 1e3,   color="crimson",    lw=2,   ls="-",
                   label=f"FP32 CUDA-core ({PEAK_FP32_TFLOPS:.1f} TFLOP/s)", zorder=1)
        ax.axhline(PEAK_TF32TC_TFLOPS * 1e3, color="darkorange", lw=1.5, ls="-.",
                   label=f"TF32-TC dense ({PEAK_TF32TC_TFLOPS:.1f} TFLOP/s)", zorder=1)
        ax.axhline(PEAK_FP16TC_TFLOPS * 1e3, color="purple",     lw=2,   ls=":",
                   label=f"FP16/BF16-TC dense ({PEAK_FP16TC_TFLOPS:.0f} TFLOP/s)", zorder=1)

        # Ridge verticals (only on full panel to avoid clutter in zoom)
        if not zoom:
            for ridge, rlabel, col in [
                (RIDGE_FP32,   f"FP32 ridge\n{RIDGE_FP32:.2f} FLOP/B",      "crimson"),
                (RIDGE_TF32TC, f"TF32-TC\n{RIDGE_TF32TC:.1f} FLOP/B",       "darkorange"),
                (RIDGE_FP16TC, f"FP16-TC\n{RIDGE_FP16TC:.0f} FLOP/B",       "purple"),
            ]:
                if x_min < ridge < x_max:
                    ax.axvline(ridge, color=col, lw=0.8, ls=":", alpha=0.4, zorder=1)
                    ax.text(ridge * 1.05, y_min * 3, rlabel,
                            color=col, fontsize=7.5, va="bottom")

        # Scatter
        for pt in all_points:
            impl  = pt["impl"]
            stage = pt["stage"]
            jx, jy = jittered.get((impl, stage), (pt["intensity"], pt["achieved_gflops"]))
            if not (x_min <= jx <= x_max and y_min <= jy <= y_max):
                continue
            color  = IMPL_COLORS.get(impl, "gray")
            marker = STAGE_MARKERS.get(stage, "o")
            ax.scatter(jx, jy, color=color, marker=marker, s=110, zorder=5,
                       edgecolors="white", linewidths=0.8)

        # Annotations — only on zoom panel where there is space
        if zoom:
            annotated = set()
            for pt in sorted(all_points, key=lambda p: -p["achieved_gflops"]):
                impl  = pt["impl"]
                stage = pt["stage"]
                jx, jy = jittered.get((impl, stage), (pt["intensity"], pt["achieved_gflops"]))
                if not (x_min <= jx <= x_max and y_min <= jy <= y_max):
                    continue
                key = (impl, stage)
                if key in annotated:
                    continue
                annotated.add(key)
                color = IMPL_COLORS.get(impl, "gray")
                short_impl = impl.replace("cublas_", "").replace("cublas", "fp32")
                ax.annotate(
                    f"{short_impl}\n{stage}",
                    xy=(jx, jy),
                    xytext=(jx * 1.25, jy * 1.35),
                    fontsize=6.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7, alpha=0.7),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"),
                    zorder=6,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", alpha=0.18)
        ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
        ax.set_ylabel("Achieved Throughput (GFLOPs/s)", fontsize=11)

    # ── Full panel ────────────────────────────────────────────────────────────
    draw_panel(ax_full, x_min=0.05, x_max=RIDGE_FP16TC * 3,
               y_min=0.5, y_max=PEAK_FP16TC_TFLOPS * 1e3 * 2.5, zoom=False)
    ax_full.set_title(
        f"Roofline — full view\n{GPU_NAME}  ·  pos={POS}  ·  batch=1",
        fontsize=11,
    )

    # ── Zoom panel ────────────────────────────────────────────────────────────
    # AI range: 0.15 – 1.5 covers all measured points; GFLOPs: 0.3 – 2 500
    draw_panel(ax_zoom, x_min=0.15, x_max=1.8,
               y_min=0.3, y_max=2500, zoom=True)
    # Draw a light box on the full panel showing the zoom region
    rect = plt.Rectangle((0.15, 0.3), 1.65, 2499.7,
                          linewidth=1.2, edgecolor="black", facecolor="none",
                          linestyle="--", zorder=7, transform=ax_full.transData)
    ax_full.add_patch(rect)
    ax_zoom.set_title(
        "Zoom: measured data (AI 0.15–1.8, GFLOPs 0.3–2500)\njitter applied to separate overlapping variants",
        fontsize=10,
    )

    # ── Shared legends ────────────────────────────────────────────────────────
    # Ceiling lines legend — below the full panel
    line_handles = [
        plt.Line2D([0], [0], color="steelblue", lw=2, ls="--",
                   label=f"Memory BW ceiling ({PEAK_BW_GBs:.0f} GB/s)"),
        plt.Line2D([0], [0], color="crimson",    lw=2, ls="-",
                   label=f"FP32 CUDA-core ({PEAK_FP32_TFLOPS:.1f} TFLOP/s)"),
        plt.Line2D([0], [0], color="darkorange",lw=1.5, ls="-.",
                   label=f"TF32-TC dense ({PEAK_TF32TC_TFLOPS:.1f} TFLOP/s)"),
        plt.Line2D([0], [0], color="purple",     lw=2, ls=":",
                   label=f"FP16/BF16-TC dense ({PEAK_FP16TC_TFLOPS:.0f} TFLOP/s)"),
    ]
    fig.legend(handles=line_handles, loc="lower left",
               bbox_to_anchor=(0.01, 0.01), fontsize=8.5,
               title="Roofline ceilings", title_fontsize=9,
               ncol=2, framealpha=0.9)

    # Impl colour legend — below the zoom panel
    impl_handles = [
        mpatches.Patch(color=IMPL_COLORS[impl], label=impl)
        for impl in ALL_IMPLS if impl in {pt["impl"] for pt in all_points}
    ]
    fig.legend(handles=impl_handles, loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=8.5,
               title="Implementation (colour)", title_fontsize=9,
               ncol=3, framealpha=0.9)

    # Stage marker legend — bottom centre
    stage_handles = [
        plt.Line2D([0], [0], marker=mk, color="gray", ls="None",
                   markersize=8, label=stage)
        for stage, mk in STAGE_MARKERS.items()
    ]
    fig.legend(handles=stage_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), fontsize=8,
               title="Stage (marker)", title_fontsize=9,
               ncol=4, framealpha=0.9)

    fig.suptitle(
        f"Llama2 7B — Roofline Analysis ({GPU_NAME})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Roofline analysis for Llama2 7B CUDA variants.")
    p.add_argument("--results-root", default="profiling_runs")
    p.add_argument("--output-dir",   default="docs/profiling_plots")
    p.add_argument("--pos", type=int, default=POS,
                   help=f"Sequence position used for attention FLOP estimates (default: {POS})")
    p.add_argument("--no-plot", action="store_true", help="Print table only, no PNG")
    args = p.parse_args()

    results_root = Path(args.results_root)
    output_dir   = Path(args.output_dir)

    # ── load stats ────────────────────────────────────────────────────────────
    all_points = []
    impls_found = []
    for impl in ALL_IMPLS:
        stats_path = results_root / impl / f"profile_stats_{impl}.csv"
        if not stats_path.exists():
            print(f"  [skip] {stats_path} not found")
            continue
        stats = load_stats_csv(stats_path)
        if not stats:
            continue
        points = compute_points(impl, stats, pos=args.pos)
        if points:
            all_points.extend(points)
            impls_found.append(impl)

    if not all_points:
        print(
            f"No profile_stats_*.csv files found under '{results_root}'.\n"
            "Run profile_stats.py (or the shell script) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded data for: {', '.join(impls_found)}")

    # ── text table (Step 4c) ──────────────────────────────────────────────────
    all_points.sort(key=lambda pt: (ALL_IMPLS.index(pt["impl"]) if pt["impl"] in ALL_IMPLS else 99,
                                    list(STAGE_MARKERS).index(pt["stage"]) if pt["stage"] in STAGE_MARKERS else 99))
    print_table(all_points)

    # ── attainment CSV ────────────────────────────────────────────────────────
    att_csv = output_dir / "roofline_attainment.csv"
    att_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(att_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "impl", "stage", "mean_ms", "flops", "byte_traffic",
            "intensity_flops_per_byte", "achieved_gflops",
            "bw_ceiling_gflops", "peak_compute_gflops",
            "bw_attainment_pct", "compute_attainment_pct",
        ])
        for pt in all_points:
            w.writerow([
                pt["impl"], pt["stage"],
                f"{pt['mean_ms']:.6f}",
                f"{pt['flops']:.0f}",
                f"{pt['byte_traffic']:.0f}",
                f"{pt['intensity']:.4f}",
                f"{pt['achieved_gflops']:.4f}",
                f"{pt['bw_ceiling_gflops']:.4f}",
                f"{pt['peak_gflops']:.4f}",
                f"{pt['bw_attainment_pct']:.2f}",
                f"{pt['cmp_attainment_pct']:.4f}",
            ])
    print(f"\nAttainment CSV saved: {att_csv}")

    # ── plot (Steps 4b) ───────────────────────────────────────────────────────
    if not args.no_plot:
        plot_roofline(all_points, output_dir / "roofline.png")


if __name__ == "__main__":
    main()
