#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_profiling_experiment.sh [options]

Options:
  -i, --impl        CUDA implementation to run: flash, cublas, tiled, or all (default: all)
  -n, --runs        Number of runs per implementation (default: 30)
  -m, --model       Model to profile (default: llama2_7b.bin)
  -p, --prompt      Prompt to use for every run (default: Once upon a time)
  -s, --steps       Number of generation steps (default: 256)
  -t, --temperature Sampling temperature (default: 1.0)
  --topp            Top-p sampling value (default: 0.9)
  --seed            Base random seed (default: 42)
  --profile-pos     Token position where profiling is triggered (default: 10)
  --results-root    Output directory root for collected runs (default: profiling_runs)
  --modal-cmd       Modal CLI command to use (default: auto-detect modal, then modal.exe)
  --python-cmd      Python command to use for stats (default: auto-detect python, python3, py -3)
  -h, --help        Show this help message

Examples:
  ./run_profiling_experiment.sh --impl flash --runs 30
  ./run_profiling_experiment.sh --impl all --runs 50 --model llama2_7b.bin
EOF
}

impl_choice="all"
runs=30
model="llama2_7b.bin"
prompt="Once upon a time"
steps=256
temperature=1.0
topp=0.9
seed=42
profile_pos=10
results_root="profiling_runs"
modal_cmd=""
python_cmd=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--impl)
      impl_choice="$2"
      shift 2
      ;;
    -n|--runs)
      runs="$2"
      shift 2
      ;;
    -m|--model)
      model="$2"
      shift 2
      ;;
    -p|--prompt)
      prompt="$2"
      shift 2
      ;;
    -s|--steps)
      steps="$2"
      shift 2
      ;;
    -t|--temperature)
      temperature="$2"
      shift 2
      ;;
    --topp)
      topp="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --profile-pos)
      profile_pos="$2"
      shift 2
      ;;
    --results-root)
      results_root="$2"
      shift 2
      ;;
    --modal-cmd)
      modal_cmd="$2"
      shift 2
      ;;
    --python-cmd)
      python_cmd="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "$runs" =~ ^[0-9]+$ ]] || [[ "$runs" -lt 1 ]]; then
  echo "Error: --runs must be a positive integer" >&2
  exit 1
fi

valid_impls=(flash cublas tiled)
if [[ "$impl_choice" == "all" ]]; then
  impls=("${valid_impls[@]}")
else
  impls=("$impl_choice")
fi

mkdir -p "$results_root"

# Resolve results root and tool paths to avoid cwd-related path issues.
if [[ "$results_root" != /* ]]; then
  results_root="$script_dir/$results_root"
fi

if [[ -z "$modal_cmd" ]]; then
  if command -v modal >/dev/null 2>&1; then
    modal_cmd="modal"
  elif command -v modal.exe >/dev/null 2>&1; then
    modal_cmd="modal.exe"
  else
    echo "Error: could not find Modal CLI. Set --modal-cmd or install Modal CLI." >&2
    exit 1
  fi
fi

if [[ -z "$python_cmd" ]]; then
  if command -v python >/dev/null 2>&1; then
    python_cmd="python"
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  elif command -v py >/dev/null 2>&1; then
    python_cmd="py -3"
  else
    echo "Error: could not find Python. Set --python-cmd or install Python/py launcher." >&2
    exit 1
  fi
fi

to_windows_path() {
  local input_path="$1"

  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$input_path"
    return
  fi

  if [[ "$input_path" =~ ^/mnt/([a-zA-Z])/(.*)$ ]]; then
    local drive="${BASH_REMATCH[1],,}"
    local rest="${BASH_REMATCH[2]}"
    printf '%s\\%s\n' "${drive^}:" "${rest//\//\\}"
    return
  fi

  printf '%s\n' "$input_path"
}

missing_csv_runs=()
saved_csv_runs=()

for impl in "${impls[@]}"; do
  case "$impl" in
    flash|cublas|tiled)
      ;;
    *)
      echo "Error: invalid implementation '$impl'" >&2
      exit 1
      ;;
  esac

  impl_dir="$results_root/$impl"
  mkdir -p "$impl_dir"

  echo "============================================================"
  echo "Running $impl for $runs iteration(s)"
  echo "Results directory: $impl_dir"
  echo "============================================================"

  for run_index in $(seq 1 "$runs"); do
    run_tag=$(printf 'run_%03d' "$run_index")
    run_dir="$impl_dir/$run_tag"
    mkdir -p "$run_dir"

    modal_app_path="$script_dir/modal_app.py"
    modal_download_dir="$run_dir"
    if [[ "$modal_cmd" == *modal.exe ]]; then
      modal_app_path="$(to_windows_path "$modal_app_path")"
      modal_download_dir="$(to_windows_path "$modal_download_dir")"
    fi

    invocation_tag="$(date +%s)_$$"

    output_file_remote="profiling_${impl}_${run_tag}_${invocation_tag}.txt"
    profile_csv="/cache/profiling_${impl}_${run_tag}_${invocation_tag}.csv"
    local_txt="$run_dir/$output_file_remote"
    local_csv="$run_dir/$(basename "$profile_csv")"
    output_file_final="profiling_${impl}_${run_tag}.txt"
    output_csv_final="profiling_${impl}_${run_tag}.csv"

    rm -f "$local_txt" "$local_csv"

    echo "--- $impl / $run_tag ---"
    "$modal_cmd" run "$modal_app_path" \
      --cuda-impl "$impl" \
      --model "$model" \
      --prompt "$prompt" \
      --steps "$steps" \
      --temperature "$temperature" \
      --topp "$topp" \
      --seed "$seed" \
      --output-file "$output_file_remote" \
      --profile-enable \
      --profile-pos "$profile_pos" \
      --profile-csv "$profile_csv" \
      --download-dir "$modal_download_dir"

    if [[ ! -f "$local_txt" ]]; then
      echo "Error: missing downloaded text log '$local_txt'" >&2
      exit 1
    fi

    # Always finalize the text log name, even if CSV download fails.
    run_txt="$run_dir/$output_file_final"
    mv -f "$local_txt" "$run_txt"

    run_csv="$run_dir/$output_csv_final"
    if [[ -f "$local_csv" ]]; then
      mv -f "$local_csv" "$run_csv"
      $python_cmd "$script_dir/profile_stats.py" --input "$run_csv" --output "$run_dir/profile_stats_${impl}.csv"
      saved_csv_runs+=("$impl/$run_tag")
      echo "Saved: $run_csv"
      echo "Saved: $run_dir/profile_stats_${impl}.csv"
    else
      missing_csv_runs+=("$impl/$run_tag")
      echo "Warning: missing downloaded profiling CSV '$local_csv'" >&2
      echo "Skipping stats generation for $impl / $run_tag" >&2
    fi

    echo "Saved: $run_txt"
  done
done

echo "============================================================"
echo "Run Summary"
echo "============================================================"
if [[ ${#saved_csv_runs[@]} -gt 0 ]]; then
  echo "Runs with CSV + stats saved (${#saved_csv_runs[@]}):"
  for run_id in "${saved_csv_runs[@]}"; do
    echo "  - $run_id"
  done
fi

if [[ ${#missing_csv_runs[@]} -gt 0 ]]; then
  echo "Runs missing CSV (${#missing_csv_runs[@]}):"
  for run_id in "${missing_csv_runs[@]}"; do
    echo "  - $run_id"
  done
else
  echo "All runs produced CSV and stats."
fi

echo "All profiling runs completed."