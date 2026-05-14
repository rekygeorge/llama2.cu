"""
investigate_fp16tc_eos.py
=========================
Tests whether cublas_fp16tc's early EOS at ~token 79 is a consistent artifact
or prompt-dependent.

Runs cublas (FP32 reference) and cublas_fp16tc side-by-side on six diverse
prompts and reports: word count, first 80 chars of output, and early-EOS flag.

Usage:
    python investigate_fp16tc_eos.py --model llama2_7b.bin
"""

import argparse
import os
import re
import subprocess
import sys

MODEL    = "llama2_7b.bin"
STEPS    = 200
SEED     = 42
IMPLS    = ["cublas", "cublas_fp16tc"]
EARLY_EOS_THRESHOLD = 130  # fewer words than this = early EOS suspected

PROMPTS = [
    "Once upon a time",                                 # original prompt
    "The knight raised his sword and",                  # mid-sentence action
    "In a galaxy far away, scientists discovered",      # sci-fi
    "She opened the old wooden door and",               # mystery/suspense start
    "The recipe called for three cups of flour and",    # mundane/practical
    "Deep in the forest, a young fox",                  # fairy-tale variant
]


def run_modal_greedy(impl, prompt, steps, seed, model):
    cmd = [
        "modal", "run", "modal_app.py",
        "--cuda-impl",   impl,
        "--model",       model,
        "--prompt",      prompt,
        "--steps",       str(steps),
        "--temperature", "0.0",
        "--seed",        str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            env=env, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", file=sys.stderr)
        return ""
    if r.returncode != 0:
        print(f"    exit {r.returncode}", file=sys.stderr)
        print(r.stderr[-500:], file=sys.stderr)
    return r.stdout


def extract_generation(raw, prompt):
    SEP = "=" * 60
    HEADER_MARKER = "🤖 LLAMA2 CUDA OUTPUT"
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
        binary_output = raw

    binary_output = re.sub(r'Using argmax sampling -> token \d+\r?\n?', '', binary_output)

    def _strip_timing(text):
        for marker in ("tok/s", "achieved", "tokens/s"):
            m = text.rfind(marker)
            if m > 0:
                ls = text.rfind("\n", 0, m)
                text = text[:ls] if ls >= 0 else text[:m]
                break
        return text.strip()

    probe = prompt[:30]
    ipt_idx = binary_output.find("Initial prompt tokens:")
    if ipt_idx >= 0:
        line_end = binary_output.find("\n", ipt_idx)
        story_region = binary_output[line_end + 1:] if line_end >= 0 else binary_output[ipt_idx:]
        pi = story_region.find(probe)
        if pi >= 0:
            return _strip_timing(story_region[pi + len(prompt):])

    start = 0
    while True:
        idx = binary_output.find(probe, start)
        if idx < 0:
            break
        if idx > 0 and binary_output[idx - 1] == '"':
            start = idx + 1
            continue
        return _strip_timing(binary_output[idx + len(prompt):])

    lines = [l for l in binary_output.split("\n")
             if l.strip() and not any(m in l for m in ("tok/s", "achieved", "tokens/s"))]
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default=MODEL)
    ap.add_argument("--steps",  type=int, default=STEPS)
    ap.add_argument("--seed",   type=int, default=SEED)
    args = ap.parse_args()

    col_w = 18
    print(f"\n{'='*80}")
    print(f"  fp16tc early-EOS investigation  |  model={args.model}  steps={args.steps}  seed={args.seed}")
    print(f"  Early EOS threshold: < {EARLY_EOS_THRESHOLD} words")
    print(f"{'='*80}\n")

    results = []   # (prompt, impl, n_words, early, snippet)

    for pidx, prompt in enumerate(PROMPTS, 1):
        print(f"Prompt {pidx}/{len(PROMPTS)}: \"{prompt}\"")
        row = {"prompt": prompt}
        for impl in IMPLS:
            print(f"  [{impl}] running ...", flush=True)
            raw  = run_modal_greedy(impl, prompt, args.steps, args.seed, args.model)
            text = extract_generation(raw, prompt)
            n    = len(text.split())
            early = n < EARLY_EOS_THRESHOLD
            snippet = text.replace("\n", " ")[:80]
            row[impl] = {"n_words": n, "early_eos": early, "snippet": snippet}
            flag = "  *** EARLY EOS ***" if early else ""
            print(f"    → {n:3d} words{flag}")
            print(f"       \"{snippet}\"")
        results.append(row)
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    hdr = f"  {'Prompt':40s}  {'cublas':>8}  {'fp16tc':>8}  {'early?':>8}"
    print(hdr)
    print(f"  {'-'*68}")

    early_count = 0
    for row in results:
        p      = row["prompt"][:40]
        ref_n  = row["cublas"]["n_words"]
        tc_n   = row["cublas_fp16tc"]["n_words"]
        early  = row["cublas_fp16tc"]["early_eos"]
        if early:
            early_count += 1
        flag   = "<EOS" if early else "ok"
        print(f"  {p:40s}  {ref_n:>8d}  {tc_n:>8d}  {flag:>8s}")

    print(f"\n  fp16tc early EOS: {early_count}/{len(PROMPTS)} prompts")
    if early_count == len(PROMPTS):
        print("  → CONSISTENT ARTIFACT: fp16tc hits early EOS on every prompt tested.")
        print("    Likely caused by accumulated FP16 rounding error in COMPUTE_16F")
        print("    tensor-core accumulation shifting logit distributions toward EOS.")
    elif early_count == 0:
        print("  → PROMPT-DEPENDENT: fp16tc did NOT hit early EOS on any tested prompt.")
        print("    The original result may have been a fluke of 'Once upon a time'.")
    else:
        print(f"  → PARTIALLY PROMPT-DEPENDENT: early EOS on {early_count}/{len(PROMPTS)} prompts.")
        print("    Certain prompt distributions are more sensitive to FP16 logit drift.")
    print()


if __name__ == "__main__":
    main()
