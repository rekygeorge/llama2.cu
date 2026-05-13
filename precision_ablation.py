#!/usr/bin/env python3
"""
precision_ablation.py — Quality vs Speed ablation for Llama2-7B CUDA variants.

Compares FP16 and BF16 variants against the FP32 cuBLAS baseline to quantify
any quality degradation introduced by reduced-precision weights.

Three measurement modes (selected by --mode):

  token-agreement  (default — no CUDA source changes required)
      Runs each variant in greedy mode (temperature=0, fixed seed) via Modal
      and compares the generated token sequence against the FP32 baseline.
      Reports: exact_match_pct, first_divergence token index, char edit_distance.

  logit-mse  (requires --logit-dir with pre-computed .npy logit dump files)
      Loads per-step logit vectors and computes MSE, max absolute error,
      and mean KL divergence vs FP32. Logit dumps are produced by the CUDA
      binaries when compiled with -DDUMP_LOGITS and run with the -L <path>
      flag (see docs/research_directions.md Step 3b for the CUDA change).

  perplexity  (requires logit dumps + --token-ids-file)
      Computes token-level cross-entropy on the stored logit vectors and
      exponentiates to get perplexity.

Usage:
    # Token agreement — works immediately, no CUDA changes needed:
    python precision_ablation.py
    python precision_ablation.py --steps 200 --impls cublas cublas_fp16 cublas_bf16

    # Logit MSE from pre-computed dump files:
    python precision_ablation.py --mode logit-mse --logit-dir ./logit_dumps

    # Full ablation (all three metrics):
    python precision_ablation.py --mode all \\
        --logit-dir ./logit_dumps \\
        --token-ids-file ./logit_dumps/token_ids.json

    # Skip Modal (re-use cached generation outputs):
    python precision_ablation.py --mode token-agreement \\
        --generations-file ./logit_dumps/generations.json

Output:
    Prints a summary table to stdout.
    Saves docs/profiling_plots/precision_ablation.csv

Token-agreement interpretation:
    100 %  → identical output to FP32;  negligible quality degradation confirmed.
    > 95 % → typical range for well-behaved reduced-precision inference.
    < 90 % → indicates the reduced-precision path is drifting from FP32 quality.

Logit-MSE interpretation (typical values for weight-quantised LLMs):
    MSE < 0.01     → negligible numerical difference
    Max abs < 0.5  → activations stay in the same numerical range
    KL  < 0.001    → probability distributions are essentially identical
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── Known throughput numbers (from profiling runs on A100) ─────────────────────

KNOWN_TOK_S = {
    "cublas":        31.06,
    "flash":         13.50,
    "tiled":          6.57,
    "cublas_fp16":   66.15,
    "cublas_bf16":   67.98,
    "cublas_fp16tc": 61.00,
}

ALL_IMPLS  = list(KNOWN_TOK_S.keys())
BASELINE   = "cublas"       # FP32 reference
VOCAB_SIZE = 32000          # Llama2 tokenizer vocabulary size

# ── Modal runner ───────────────────────────────────────────────────────────────

def run_modal_greedy(impl, steps, prompt, seed, model, modal_cmd="modal"):
    """Run one greedy (temperature=0, deterministic) Modal inference.
    Returns raw stdout string."""
    cmd = [
        modal_cmd, "run", "modal_app.py",
        "--cuda-impl",   impl,
        "--model",       model,
        "--prompt",      prompt,
        "--steps",       str(steps),
        "--temperature", "0.0",
        "--seed",        str(seed),
    ]
    print(f"  [{impl}] running {steps} greedy steps ...", flush=True)
    import os
    child_env = os.environ.copy()
    # Force the Modal CLI's Python interpreter to use UTF-8 for stdout/stderr.
    # Without this, on Windows the child process inherits the cp1252 pipe
    # encoding and crashes when Modal prints Unicode glyphs such as ✓.
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=child_env, timeout=600
        )
    except subprocess.TimeoutExpired:
        print(f"  [{impl}] TIMEOUT after 600 s", file=sys.stderr)
        return ""
    if result.returncode != 0:
        print(
            f"  [{impl}] exit {result.returncode}; stderr tail:",
            file=sys.stderr,
        )
        print(result.stderr[-1000:], file=sys.stderr)
    return result.stdout


def run_modal_with_logit_dump(impl, steps, prompt, seed, model, logit_dir, modal_cmd="modal"):
    """Trigger a Modal run with -DDUMP_LOGITS and download the logit dump file.

    Writes <logit_dir>/<impl>.bin (raw float32 logit vectors, one per generation
    step) and, for the cublas baseline, <logit_dir>/cublas_token_ids.json.
    Returns True on success.
    """
    logit_dir = Path(logit_dir)
    cmd = [
        modal_cmd, "run", "modal_app.py",
        "--cuda-impl",   impl,
        "--model",       model,
        "--prompt",      prompt,
        "--steps",       str(steps),
        "--temperature", "0.0",
        "--seed",        str(seed),
        "--dump-logits",
    ]
    print(f"  [{impl}] running logit dump ({steps} greedy steps) ...", flush=True)
    import os
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=child_env, timeout=1200
        )
    except subprocess.TimeoutExpired:
        print(f"  [{impl}] dump TIMEOUT after 1200 s", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(f"  [{impl}] dump failed (exit {result.returncode})", file=sys.stderr)
        print(result.stderr[-1000:], file=sys.stderr)
        return False

    # modal_app.py's local entrypoint already downloads the logit bin to
    # ./logit_dumps/<impl>.bin (default download_dir=".", remote="logit_dumps/<impl>.bin").
    # Check both the requested logit_dir and the fixed default location.
    logit_dir.mkdir(parents=True, exist_ok=True)
    local_bin = logit_dir / f"{impl}.bin"
    if not local_bin.exists():
        # Fallback: modal run deposited it at ./logit_dumps/<impl>.bin
        alt = Path("logit_dumps") / f"{impl}.bin"
        if alt.exists() and alt.resolve() != local_bin.resolve():
            import shutil
            shutil.copy2(alt, local_bin)
    if local_bin.exists():
        print(f"  [{impl}] logit dump saved -> {local_bin}")
    else:
        print(f"  [{impl}] WARNING: dump run succeeded but {local_bin} not found.",
              file=sys.stderr)
        print(f"           Check that modal_app.py downloaded to ./logit_dumps/.",
              file=sys.stderr)
        return False

    if impl == "cublas":
        local_ids = logit_dir / "cublas_token_ids.json"
        if not local_ids.exists():
            alt_ids = Path("logit_dumps") / "cublas_token_ids.json"
            if alt_ids.exists() and alt_ids.resolve() != local_ids.resolve():
                import shutil
                shutil.copy2(alt_ids, local_ids)
        if local_ids.exists():
            print(f"  [cublas] token IDs saved -> {local_ids}")
        else:
            print(f"  [cublas] WARNING: token IDs not found at {local_ids}",
                  file=sys.stderr)
    return True


def extract_generation(raw, prompt):
    """Extract the model-generated continuation from Modal stdout.

    modal_app.py wraps binary stdout between ='*60 separator lines:
        ============================================================
        🤖 LLAMA2 CUDA OUTPUT (...) - Strategy N:
        ============================================================
        <binary stdout — prompt + continuation + timing>
        ============================================================
        🔍 Process completed...

    We isolate that region, locate the prompt inside it, then strip
    the trailing timing line (e.g. "achieved 31.06 tok/s").
    """
    SEP = "=" * 60
    HEADER_MARKER = "🤖 LLAMA2 CUDA OUTPUT"

    # Narrow to the binary-output region between the two separator lines
    header_idx = raw.find(HEADER_MARKER)
    if header_idx >= 0:
        after_header = raw[header_idx:]
        sep_after = after_header.find(SEP)
        if sep_after >= 0:
            binary_region = after_header[sep_after + len(SEP):]
            end_sep = binary_region.find(SEP)
            binary_output = binary_region[:end_sep].strip() if end_sep >= 0 else binary_region.strip()
        else:
            binary_output = after_header.strip()
    else:
        binary_output = raw  # Fallback: use full output

    # sample_debug() in cublas/flash variants prints "Using argmax sampling -> token N\n"
    # to stdout for every step, interleaving with story text.  Strip those lines now
    # so all downstream strategies see clean story text.
    binary_output = re.sub(r'Using argmax sampling -> token \d+\r?\n?', '', binary_output)

    def _strip_timing(text):
        """Strip trailing tok/s timing line from generated text."""
        for marker in ("tok/s", "achieved", "tokens/s"):
            m = text.rfind(marker)
            if m > 0:
                ls = text.rfind("\n", 0, m)
                text = text[:ls] if ls >= 0 else text[:m]
                break
        return text.strip()

    probe = prompt[:30]

    # Strategy 1: cublas/flash/tiled print "Initial prompt tokens: N" immediately
    # before the generated text begins.  Find that line and take everything after
    # it — this skips the debug preamble (tokenizer debug, model loading, etc.)
    # which also contains the prompt inside quoted strings like:
    #   Input text: "Once upon a time"
    ipt_idx = binary_output.find("Initial prompt tokens:")
    if ipt_idx >= 0:
        line_end = binary_output.find("\n", ipt_idx)
        story_region = binary_output[line_end + 1:] if line_end >= 0 else binary_output[ipt_idx:]
        pi = story_region.find(probe)
        if pi >= 0:
            return _strip_timing(story_region[pi + len(prompt):])

    # Strategy 2: fp16/bf16/fp16tc — all CUDA init goes to stderr so
    # binary_output IS just the generated text.  Find the prompt not preceded
    # by a quote (avoids hitting debug lines like:  Input text: "Once upon a time").
    start = 0
    while True:
        idx = binary_output.find(probe, start)
        if idx < 0:
            break
        if idx > 0 and binary_output[idx - 1] == '"':
            start = idx + 1
            continue
        return _strip_timing(binary_output[idx + len(prompt):])

    # Fallback: return non-timing lines
    lines = [l for l in binary_output.split("\n")
             if l.strip() and not any(m in l for m in ("tok/s", "achieved", "tokens/s"))]
    return "\n".join(lines).strip()


# ── Token-agreement metrics ────────────────────────────────────────────────────

def word_tokens(text):
    """Split on whitespace; lowercase for case-insensitive comparison."""
    return text.lower().split()


def compute_token_agreement(ref_tokens, cmp_tokens):
    """Return (exact_match_pct, first_divergence_idx, n_deviations).

    Comparison is over min(len(ref), len(cmp)) shared positions."""
    L = min(len(ref_tokens), len(cmp_tokens))
    if L == 0:
        return 0.0, 0, 0
    matches = sum(1 for a, b in zip(ref_tokens, cmp_tokens) if a == b)
    first_div = next(
        (i for i, (a, b) in enumerate(zip(ref_tokens, cmp_tokens)) if a != b),
        L,
    )
    return 100.0 * matches / L, first_div, L - matches


def levenshtein_chars(a, b, cap=3000):
    """Character-level Levenshtein edit distance, capped at `cap` chars."""
    a, b = a[:cap], b[:cap]
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * n
        for j, cb in enumerate(b):
            curr[j + 1] = (
                prev[j] if ca == cb
                else 1 + min(prev[j], prev[j + 1], curr[j])
            )
        prev = curr
    return prev[n]


# ── Logit-file helpers ─────────────────────────────────────────────────────────

def load_logits(path: Path):
    """Load logit dump: returns np.ndarray of shape (N_steps, VOCAB_SIZE).

    Supports .npy files (preferred) and raw float32 binary files."""
    if not HAS_NUMPY:
        raise RuntimeError(
            "numpy is required for logit-mse / perplexity modes.\n"
            "Install with:  pip install numpy"
        )
    if path.suffix == ".npy":
        return np.load(str(path))          # shape (N, VOCAB_SIZE)
    # Raw binary float32
    raw = np.frombuffer(path.read_bytes(), dtype=np.float32)
    n_steps = raw.size // VOCAB_SIZE
    return raw[: n_steps * VOCAB_SIZE].reshape(n_steps, VOCAB_SIZE)


def find_logit_file(logit_dir: Path, impl: str):
    """Search common naming conventions for a logit dump file."""
    for name in (
        f"{impl}.npy", f"logits_{impl}.npy",
        f"{impl}.bin", f"logits_{impl}.bin",
    ):
        p = logit_dir / name
        if p.exists():
            return p
    return None


def compute_logit_mse(ref_logits, cmp_logits):
    """Return (mse, max_abs_error, mean_kl_divergence) between two logit arrays.

    KL divergence is computed as KL(p_ref || p_cmp) where p = softmax(logits).
    A small KL (< 0.001) means the token probability distributions are
    essentially identical despite numerical differences in the raw logits."""
    steps = min(len(ref_logits), len(cmp_logits))
    ref = ref_logits[:steps].astype(np.float64)
    cmp = cmp_logits[:steps].astype(np.float64)
    diff = ref - cmp
    mse     = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))

    def softmax(x):
        ex = np.exp(x - x.max(axis=-1, keepdims=True))
        return ex / ex.sum(axis=-1, keepdims=True)

    p_ref = softmax(ref)
    p_cmp = softmax(cmp)
    kl = np.sum(
        p_ref * (np.log(p_ref + 1e-12) - np.log(p_cmp + 1e-12)),
        axis=-1,
    )
    return mse, max_abs, float(np.mean(kl))


def compute_perplexity(logits, target_token_ids):
    """Token-level perplexity of a fixed sequence under the model's distribution.

    logits           : np.ndarray (N_steps, VOCAB_SIZE) — raw pre-softmax logits
    target_token_ids : list[int] — ground-truth next-token IDs, length = N_steps

    Perplexity = exp( -mean( log P(target_t | context_{<t}) ) )

    Lower is better; a model matching FP32 will have ~identical perplexity."""
    steps = min(len(logits), len(target_token_ids))
    lgt = logits[:steps].astype(np.float64)
    # numerically stable log-softmax
    lgt -= lgt.max(axis=-1, keepdims=True)
    log_probs = lgt - np.log(np.sum(np.exp(lgt), axis=-1, keepdims=True))
    targets = np.array(target_token_ids[:steps], dtype=int)
    nll = -np.mean(log_probs[np.arange(steps), targets])
    return float(np.exp(nll))


# ── Table formatting ───────────────────────────────────────────────────────────

COL_W = {
    "impl":       16,
    "tok_s":       7,
    "speedup":     7,
    "tok_agree":  11,
    "first_div":   9,
    "edit_dist":  10,
    "logit_mse":  11,
    "max_abs":    10,
    "kl_div":      9,
    "perplexity": 11,
}

def _fmt(v, width):
    return str(v).rjust(width)

def print_header():
    print(
        f"{'impl':<16}"
        f"{'tok/s':>7} {'speedup':>7}"
        f"{'tok_agree%':>11} {'1st_div':>9} {'edit_dist':>10}"
        f"{'logit_mse':>11} {'max_abs':>10} {'kl_div':>9}"
        f"{'perplexity':>11}"
    )

def print_row(r):
    print(
        f"{r['impl']:<16}"
        f"{str(r['tok_s']):>7} {str(r['speedup']):>7}"
        f"{str(r['tok_agree']):>11} {str(r['first_div']):>9} {str(r['edit_dist']):>10}"
        f"{str(r['logit_mse']):>11} {str(r['max_abs']):>10} {str(r['kl_div']):>9}"
        f"{str(r['perplexity']):>11}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        choices=["token-agreement", "logit-mse", "perplexity", "all"],
        default="token-agreement",
        help="Which quality metrics to compute (default: token-agreement)",
    )
    ap.add_argument(
        "--impls", nargs="+", default=ALL_IMPLS,
        help=f"Implementations to evaluate (default: all {len(ALL_IMPLS)})",
    )
    ap.add_argument("--steps",  type=int, default=200,
                    help="Number of greedy generation steps (default: 200)")
    ap.add_argument("--prompt", default="Once upon a time",
                    help="Generation prompt (must be identical across all runs)")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--model",  default="llama2_7b.bin")
    ap.add_argument("--modal-cmd", default="modal",
                    help="Modal CLI executable (e.g. 'modal.exe' on Windows)")
    ap.add_argument(
        "--logit-dir", type=Path, default=None,
        help="Directory containing <impl>.npy logit dump files "
             "(required for --mode logit-mse / perplexity / all)",
    )
    ap.add_argument(
        "--auto-dump-logits", action="store_true", default=False,
        help="Auto-trigger Modal dump runs for missing logit files "
             "(requires -DDUMP_LOGITS support in CUDA sources; default: off).",
    )
    ap.add_argument(
        "--token-ids-file", type=Path, default=None,
        help="JSON file with ground-truth next-token IDs for perplexity "
             "(list of ints, length = steps). Produced by the FP32 baseline "
             "run when compiled with -DDUMP_LOGITS.",
    )
    ap.add_argument(
        "--generations-file", type=Path, default=None,
        help="JSON file {impl: generated_text} to skip Modal runs and re-use "
             "previously captured outputs for token-agreement mode.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("docs/profiling_plots"),
        help="Directory for output CSV (default: docs/profiling_plots/)",
    )
    ap.add_argument(
        "--save-generations", type=Path, default=None,
        help="Save captured generation strings to this JSON file for re-use.",
    )
    args = ap.parse_args()

    do_token = args.mode in ("token-agreement", "all")
    do_logit = args.mode in ("logit-mse", "all")
    do_ppl   = args.mode in ("perplexity", "all")

    if (do_logit or do_ppl) and args.logit_dir is None:
        if args.auto_dump_logits:
            args.logit_dir = Path("./logit_dumps")
            print(f"  --logit-dir not set; will use {args.logit_dir} (auto-created)")
        else:
            ap.error(
                "--logit-dir is required for logit-mse / perplexity modes.\n"
                "Pass --auto-dump-logits to generate logit dumps automatically,\n"
                "or generate manually: compile CUDA binary with -DDUMP_LOGITS\n"
                "and run with -L <output_dir>.\n"
                "See docs/precision_ablation.md for details."
            )

    # ── 1. Token-agreement ────────────────────────────────────────────────────
    generations = {}

    if do_token:
        if args.generations_file and args.generations_file.exists():
            print(f"Loading cached generations from {args.generations_file}")
            with open(args.generations_file) as f:
                generations = json.load(f)
        else:
            print(
                f"\n=== Token Agreement (temperature=0, seed={args.seed}, "
                f"steps={args.steps}, model={args.model}) ==="
            )
            for impl in args.impls:
                print(f"\n[{impl}]")
                raw = run_modal_greedy(
                    impl, args.steps, args.prompt, args.seed,
                    args.model, args.modal_cmd,
                )
                generations[impl] = extract_generation(raw, args.prompt)
                n_words = len(generations[impl].split())
                print(f"  → captured {n_words} words")

            if args.save_generations:
                args.save_generations.parent.mkdir(parents=True, exist_ok=True)
                with open(args.save_generations, "w") as f:
                    json.dump(generations, f, indent=2)
                print(f"\nGenerations saved: {args.save_generations}")

    # ── 2. Logit files ────────────────────────────────────────────────────────
    logit_data = {}

    if do_logit or do_ppl:
        if not HAS_NUMPY:
            print(
                "ERROR: numpy is required for logit-mse / perplexity modes.\n"
                "Install with:  pip install numpy",
                file=sys.stderr,
            )
            sys.exit(1)
        for impl in args.impls:
            p = find_logit_file(args.logit_dir, impl)
            if p is None and getattr(args, "auto_dump_logits", False):
                print(f"  [{impl}] logit file missing — triggering dump run ...")
                ok = run_modal_with_logit_dump(
                    impl, args.steps, args.prompt, args.seed,
                    args.model, args.logit_dir, args.modal_cmd,
                )
                if ok:
                    p = find_logit_file(args.logit_dir, impl)
            if p:
                print(f"Loading logits [{impl}]: {p}")
                logit_data[impl] = load_logits(p)
            else:
                print(f"  [skip] no logit file for '{impl}' in {args.logit_dir}")

    token_ids = None
    if do_ppl:
        token_ids_path = args.token_ids_file
        # Auto-detect: if cublas ran a dump, cublas_token_ids.json is in logit_dir
        if (token_ids_path is None or not token_ids_path.exists()) and args.logit_dir:
            candidate = args.logit_dir / "cublas_token_ids.json"
            if candidate.exists():
                token_ids_path = candidate
                print(f"Auto-detected token IDs: {token_ids_path}")
        if token_ids_path and token_ids_path.exists():
            with open(token_ids_path) as f:
                token_ids = json.load(f)
            print(f"Loaded {len(token_ids)} ground-truth token IDs")
        else:
            print(
                "  NOTE: --token-ids-file not provided or not found; "
                "perplexity will be skipped."
            )
            do_ppl = False

    # ── 3. Build results ──────────────────────────────────────────────────────
    ref_gen    = generations.get(BASELINE, "")
    ref_tokens = word_tokens(ref_gen)
    ref_logits = logit_data.get(BASELINE)

    results = []
    for impl in args.impls:
        speedup = KNOWN_TOK_S.get(impl, 0) / KNOWN_TOK_S.get(BASELINE, 1)

        row = {
            "impl":       impl,
            "tok_s":      f"{KNOWN_TOK_S.get(impl, ''):.2f}" if impl in KNOWN_TOK_S else "—",
            "speedup":    f"{speedup:.2f}x" if impl in KNOWN_TOK_S else "—",
            "tok_agree":  "",
            "first_div":  "",
            "edit_dist":  "",
            "logit_mse":  "",
            "max_abs":    "",
            "kl_div":     "",
            "perplexity": "",
        }

        # Token agreement
        if do_token:
            if impl == BASELINE:
                row["tok_agree"] = "100.0 (ref)"
                row["first_div"] = "N/A"
                row["edit_dist"] = "0"
            elif impl in generations:
                cmp_tokens = word_tokens(generations[impl])
                pct, first_div, n_dev = compute_token_agreement(
                    ref_tokens, cmp_tokens
                )
                ed = levenshtein_chars(ref_gen, generations[impl])
                row["tok_agree"] = f"{pct:.1f}"
                row["first_div"] = str(first_div)
                row["edit_dist"] = str(ed)

        # Logit MSE
        if do_logit:
            if impl == BASELINE and ref_logits is not None:
                row["logit_mse"] = "0.000000"
                row["max_abs"]   = "0.0000"
                row["kl_div"]    = "0.000000"
            elif impl in logit_data and ref_logits is not None:
                mse, max_abs, kl = compute_logit_mse(ref_logits, logit_data[impl])
                row["logit_mse"] = f"{mse:.6f}"
                row["max_abs"]   = f"{max_abs:.4f}"
                row["kl_div"]    = f"{kl:.6f}"

        # Perplexity
        if do_ppl and token_ids is not None:
            if impl in logit_data:
                ppl = compute_perplexity(logit_data[impl], token_ids)
                row["perplexity"] = f"{ppl:.3f}"

        results.append(row)

    # ── 4. Print table ────────────────────────────────────────────────────────
    bar = "=" * 107
    print(f"\n{bar}")
    print("Precision Ablation — Llama2 7B on A100  (baseline = cublas FP32)")
    print(
        f"  mode={args.mode} | steps={args.steps} | seed={args.seed} "
        f"| temp=0.0 (greedy) | model={args.model}"
    )
    print(bar)
    print_header()
    print("-" * 107)
    for row in results:
        print_row(row)
    print("-" * 107)

    # Key interpretation
    print(
        "\nInterpretation:"
        "\n  tok_agree%  : ≥ 95 % — expected for weight-quantised FP16/BF16 at temperature=0"
        "\n  logit_mse   : < 0.01  — negligible numerical difference"
        "\n  kl_div      : < 0.001 — probability distributions are effectively identical"
        "\n  perplexity  : within 0.1 nats of FP32 baseline — negligible quality loss"
    )

    if do_token and BASELINE in generations and ref_gen:
        print(f"\nFP32 reference output (first 200 chars):")
        print(f"  {ref_gen[:200]!r}")

    if not do_logit:
        print(
            "\nNote: logit MSE and KL divergence were not computed.\n"
            "  Run with --mode all --logit-dir <dir> for the complete ablation.\n"
            "  Logit dumps require the CUDA binary to be compiled with -DDUMP_LOGITS\n"
            "  and run with the -L <output_dir> flag.\n"
            "  See docs/research_directions.md Part 3 Step 3b for instructions."
        )

    # ── 5. Save CSV ───────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "precision_ablation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {out_csv}")


if __name__ == "__main__":
    main()
