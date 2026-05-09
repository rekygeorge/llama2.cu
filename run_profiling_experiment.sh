#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run_profiling_experiment.sh [options]

Options:
  -i, --impl        CUDA implementation to run: flash, cublas, tiled, cublas_fp16, cublas_bf16, cublas_fp16tc, or all (default: all)
  -n, --runs        Number of runs per implementation (default: 30)
  -m, --model       Model to profile (default: llama2_7b.bin)
  -p, --prompt      Prompt to use for every run (default: Once upon a time)
  -s, --steps       Number of generation steps (default: 256)
  -t, --temperature Sampling temperature (default: 1.0)
  --topp            Top-p sampling value (default: 0.9)
  --seed            Base random seed (default: 42)
  --profile-pos     Token position where profiling is triggered (default: 50)
  --profile-pos-list  Comma-separated positions for a sequence-length sweep, e.g. "10,50,100,200,500,1024".
                      When given, --profile-pos is ignored; all positions are swept in order.
                      Steps are auto-bumped if pos >= steps to ensure the profiler fires.
  --results-root    Output directory root for collected runs (default: profiling_runs)
  --modal-cmd       Modal CLI command to use (default: auto-detect modal, then modal.exe)
  --python-cmd      Python command to use for stats (default: auto-detect python, python3, py -3)
  -h, --help        Show this help message

Examples:
  ./run_profiling_experiment.sh --impl flash --runs 30
  ./run_profiling_experiment.sh --impl cublas_fp16 --runs 10
  ./run_profiling_experiment.sh --impl all --runs 50 --model llama2_7b.bin
  ./run_profiling_experiment.sh --impl cublas --runs 5 --profile-pos-list "10,50,100,200,500"
  ./run_profiling_experiment.sh --impl all --runs 5 --profile-pos-list "50,200,500,1024"
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
profile_pos=50
profile_pos_list=""
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
    --profile-pos-list)
      profile_pos_list="$2"
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

valid_impls=(flash cublas tiled cublas_fp16 cublas_bf16 cublas_fp16tc)
if [[ "$impl_choice" == "all" ]]; then
  impls=("${valid_impls[@]}")
else
  impls=("$impl_choice")
fi

# Build the list of sequence positions to sweep.
if [[ -n "$profile_pos_list" ]]; then
  IFS=',' read -ra pos_values <<< "$profile_pos_list"
  for i in "${!pos_values[@]}"; do
    pos_values[$i]="${pos_values[$i]// /}"  # strip any whitespace
  done
else
  pos_values=("$profile_pos")
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
    flash|cublas|tiled|cublas_fp16|cublas_bf16|cublas_fp16tc)
      ;;
    *)
      echo "Error: invalid implementation '$impl'" >&2
      echo "Valid choices: flash cublas tiled cublas_fp16 cublas_bf16 cublas_fp16tc all" >&2
      exit 1
      ;;
  esac

  impl_dir="$results_root/$impl"
  mkdir -p "$impl_dir"

  echo "============================================================"
  echo "Running $impl for $runs iteration(s)"
  echo "Results directory: $impl_dir"
  echo "============================================================"

  # Accumulating CSV — all runs for this impl append to the same file.
  impl_csv="$impl_dir/profiling_${impl}.csv"

  # download-dir is the workspace root so that modal_app.py reconstructs
  # profiling_runs/{impl}/... paths flat under impl_dir.
  modal_app_path="$script_dir/modal_app.py"
  modal_download_dir="$script_dir"
  if [[ "$modal_cmd" == *modal.exe ]]; then
    modal_app_path="$(to_windows_path "$modal_app_path")"
    modal_download_dir="$(to_windows_path "$modal_download_dir")"
  fi

  for pos_val in "${pos_values[@]}"; do
    # Ensure enough generation steps to reach the profiling position.
    effective_steps="$steps"
    if (( pos_val >= steps )); then
      effective_steps=$(( pos_val + 10 ))
      echo "Note: --steps bumped to $effective_steps to reach profile-pos=$pos_val"
    fi

    for run_index in $(seq 1 "$runs"); do
      invocation_tag="$(date +%s)_$$"
      # Vary seed per run so each run exercises a different sampling path
      # while remaining reproducible: seed_0=base, seed_1=base+1, ...
      run_seed=$((seed + run_index - 1))

      # TXT gets a unique timestamp so runs never overwrite each other.
      output_file_remote="profiling_runs/${impl}/profiling_${impl}_${invocation_tag}.txt"
      # CSV is a single accumulating path — each run appends one row.
      profile_csv="/cache/profiling_runs/${impl}/profiling_${impl}.csv"
      local_txt="$impl_dir/profiling_${impl}_${invocation_tag}.txt"

      rm -f "$local_txt"

      echo "--- $impl / pos=$pos_val / run $run_index of $runs (seed=$run_seed) ---"
      "$modal_cmd" run "$modal_app_path" \
        --cuda-impl "$impl" \
        --model "$model" \
        --prompt "$prompt" \
        --steps "$effective_steps" \
        --temperature "$temperature" \
        --topp "$topp" \
        --seed "$run_seed" \
        --output-file "$output_file_remote" \
        --profile-enable \
        --profile-pos "$pos_val" \
        --profile-csv "$profile_csv" \
        --download-dir "$modal_download_dir"

      if [[ ! -f "$local_txt" ]]; then
        echo "Error: missing downloaded text log '$local_txt'" >&2
        exit 1
      fi
      echo "Saved: $local_txt"

      if [[ -f "$impl_csv" ]]; then
        saved_csv_runs+=("$impl/pos=${pos_val}/run_$(printf '%03d' "$run_index")")
      else
        missing_csv_runs+=("$impl/pos=${pos_val}/run_$(printf '%03d' "$run_index")")
        echo "Warning: profiling CSV not yet present at '$impl_csv'" >&2
      fi
    done
  done

  # Generate stats from the accumulated CSV once all runs for this impl are done.
  # Skip the first 3 rows (warmup) before computing statistics.
  if [[ -f "$impl_csv" ]]; then
    stats_csv="$impl_dir/profile_stats_${impl}.csv"
    $python_cmd "$script_dir/profile_stats.py" --input "$impl_csv" --warmup-skip 3 --output "$stats_csv"
    echo "Saved: $impl_csv"
    echo "Saved: $stats_csv"
  else
    echo "Warning: no profiling CSV found for $impl after $runs run(s)" >&2
  fi
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