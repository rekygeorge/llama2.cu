# CUDA Llama2 - Multiple CUDA Variants

🚀 **High-performance CUDA implementation of Llama2 inference with AWS S3 integration and Modal cloud deployment**

This project implements a complete CUDA-accelerated version of Andrej Karpathy's llama2.c, optimized for cloud deployment on Modal's infrastructure with automatic model caching and S3 integration.

## 🧠 Core Optimization Techniques

The repository now supports four CUDA variants selected with `--cuda-impl`:
- `cublas`: uses cuBLAS (`llama2_cublas.cu`) for the projection-heavy layers and custom CUDA kernels for attention and normalization.
- `flash`: uses the custom kernel path (`llama2_flash.cu`) for more of the linear layers and attention helpers.
- `tiled`: uses a custom shared-memory tiled matmul path (`llama2_tiled.cu`) and avoids the Flash Attention implementation.
- `cublas_fp16`: mixed-precision variant (`llama2_cublas_fp16.cu`) — weights stored as FP16 (`__half`) on the GPU, all matrix multiplications use `cublasGemmEx` with `CUBLAS_COMPUTE_32F` (FP32 accumulation). Halves GPU VRAM for weights with no quality loss vs full FP32.

All variants share the same model pipeline and most of the optimization building blocks below.

### Flash Attention Implementation
- **Multi-Head Flash Attention Kernel**: Custom CUDA attention path implemented in `multi_head_flashattention_kernel`
- **Memory-Efficient Attention**: Avoids large intermediate attention matrices by using block-wise computation
- **Tiled Computation**: Uses block sizing (`Bc`) and shared memory to fit attention work on the GPU
- **Online Softmax**: Numerically stable softmax using running max/sum statistics
- **Grouped Query Attention (GQA)**: Supports KV head sharing across query heads through `n_kv_heads`

### Advanced CUDA Optimizations in `forward_gpu()`

#### 1. **Adaptive RMSNorm (`cuda_rmsnorm_adaptive`)**
- GPU-specific kernel selection based on compute capability
- A100-optimized and RTX 2070 Super-optimized paths are both present
- Automatic hardware detection and configuration

#### 2. **2D Tiled Matrix Multiplication (`cuda_matmul_2d_tiled`)**
- Shared memory optimization with configurable tile sizes
- Used by the custom-kernel path and available as a reusable matmul helper
- Block-wise computation to maximize cache utilization
- Supports the 96KB shared memory configuration used by the app

#### 3. **Memory-Optimized FFN (`cuda_memory_optimized_ffn_w1w3`)**
- Fused W1 and W3 projections to reduce memory bandwidth
- Present as a custom FFN helper in the CUDA codebase
- Optimized for feed-forward network bottlenecks

#### 4. **RoPE (Rotary Position Embedding)**
- Parallel processing of rotation pairs
- Applied to Q and K tensors before attention
- Position-aware attention enhancement

#### 5. **KV Cache Management**
- Device-to-device memory copies for efficient caching
- Layer-wise cache organization
- Memory access pattern optimization

#### 6. **Fused Operations**
- Combined SwiGLU activation and W2 projection
- Residual connection fusion with output projection
- Reduced kernel launch overhead

### Implementation Differences

- `cublas` variant: uses `cublasSgemv` for Q/K/V, FFN, and classifier projections while keeping the Flash Attention path custom.
- `flash` variant: keeps more of the linear layers on custom CUDA helpers and is useful for comparing non-cuBLAS behavior.
- `tiled` variant: uses the custom shared-memory tiled matmul implementation in `llama2_tiled.cu` and is the best option when you want to compare tile-based matrix-vector multiplication without Flash Attention.
- `cublas_fp16` variant: uses `llama2_cublas_fp16.cu`. Weights are converted from the FP32 checkpoint to FP16 on the CPU and uploaded to the GPU as `__half`. All matmuls use `cublasGemmEx` with `(FP16, FP16) → FP32` and `CUBLAS_COMPUTE_32F`. Activations remain FP32 throughout. Requires CC ≥ 7.0 (Volta+). Supports `--verify` when compiled manually to compare GPU logits against a CPU FP16 forward pass. Approximately halves weight VRAM vs FP32.
- The performance numbers in this README are implementation-specific and should be read as mode-dependent.

### Performance Features
- **Shared Memory Configuration**: Configurable 48KB to 96KB per block
- **Cooperative Groups**: Advanced thread synchronization
- **cuBLAS Integration**: Optimized BLAS operations for large matrices
- **Dynamic Architecture Detection**: Automatic GPU compute capability detection
- **Memory Coalescing**: Optimized memory access patterns

## 📋 Table of Contents

## VLLM Inferencing 

```Python
modal run vllm_inference.py
```

### VLLM Performance Metrics

🔢 **Final Metrics Summary:**
  * Generated 512 tokens in 12.12s
  * Speed: 42.3 tokens/second
  * Total tokens processed: 564

**vLLM Runtime Findings (modal run vllm_inference.py)**

- **Model:** `meta-llama/Llama-2-7b-chat-hf` (7B parameters)
- **vLLM config:** `dtype=torch.float32`, `quantization=None`
- **Actual on-disk checkpoint:** FP16 safetensors shard in the Hugging Face cache, not a single root-level file
- **Cache location:** `/tmp/model_cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/<revision>/...`
- **First tensor inspected:** `lm_head.weight` with `dtype=torch.float16` and shape `(32000, 4096)`. This confirms the downloaded checkpoint is FP16 on disk, not FP32.
- **Model size estimate:** about 26.08 GiB if FP32, about 13.04 GiB if FP16/BF16
- **Cache size observed in run:** 50.21 GiB total across cached files and shards
- **CUDA graphs:** Graph capture completed during startup (logged as "Graph capturing finished in 10 secs") — note that CUDA graphs may add ~1–3 GiB GPU memory.
- **Observed tokens / timing:** Prompt tokens: 52; Generated tokens: 512; Total tokens: 564; Generation time: 11.96s; Tokens/second: 42.82; Finish reason: `length`.
- **Security note:** The run printed an HF token to stdout during testing. Do NOT hardcode or commit tokens. Use environment variables or Modal secrets instead.



## 📊 Performance Benchmarks

### Hardware Specifications

| Environment | GPU | VRAM | CPU | RAM |
|-------------|-----|------|-----|-----|
| **Local** | NVIDIA GeForce RTX 2070 Super | 8GB | Intel i9 | 32GB |
| **Modal** | NVIDIA A100 | 40GB | AMD EPYC | 32GB |

### Model Performance Comparison


#### Llama2 7B Model (26GB)

| Metric | Local RTX 2070 Super | Modal A100 | Improvement |
|--------|---------------------|------------|-------------|
| **Inference Speed** | Insufficient RAM | 4 tokens/sec |  |


### C vs CUDA Performance (Local RTX 2070 Super)

#### Stories15M Model

| Implementation | Tokens/Second |  Relative Performance |
|----------------|---------------|---------------------|
| **C (CPU)** | 65 tokens/sec |  Baseline |
| **CUDA (GPU)** | 360 tokens/sec |  5.5x faster |


### CUDA Performance (A100)

#### Stories15M Model

| Implementation | Tokens/Second | 
|----------------|---------------|
| **CUDA (GPU)** | 415 tokens/sec |

#### Llama2 7B Model
> **Llama2 7B Model: CUDA vs VLLM (A100 GPU)**  
>  
> | Implementation | Tokens/Second |  
> |----------------|---------------|  
> | **CUDA naive implementation (GPU)** | **4 tokens/sec** |  
> | **CUDA tiled implementation (GPU)** | **6.57 tokens/sec** |  
> | **CUDA flash implementation (GPU)** | **13.5 tokens/sec** |  
> | **CUDA with cublas (GPU)** | **31.06 tokens/sec** |  
> | **CUDA FP16 weights / cublas_fp16 (GPU)** | **66.15 tokens/sec** |  
> | **VLLM (GPU)** | 42.82 tokens/sec |  
>  
> *Note: This table highlights the direct comparison between custom CUDA and VLLM implementations on the A100 GPU. The tiled run above was measured with `--cuda-impl tiled --model llama2_7b.bin --prompt "Once upon a time" --steps 256 --temperature 1.0 --topp 0.9 --seed 42`. The `cublas_fp16` run used the same settings. The VLLM figure (42.82 tok/s) is the measured generation-only throughput from a `modal run vllm_inference.py` run on the same A100 (512 tokens, 11.96 s, excluding prompt-encoding time). `cublas_fp16` at 66 tok/s exceeds VLLM on raw single-sequence throughput.*

### Measured Profiling Snapshot

Means computed from all available profiling CSV runs for the 7B model on the Modal A100 (pos=10, layer 0).

| Implementation | Total Layer Time (ms) | Estimated 32-Layer Time (ms) | Estimated tok/s | Runs |
|----------------|----------------------:|------------------------------:|----------------:|-----:|
| `tiled`        |              3.326547 |                   106.449503 |        9.395141 |    10 |
| `flash`        |              2.609617 |                    83.507758 |       12.365712 |   10 |
| `cublas`       |              0.926208 |                    29.638656 |       33.740235 |   10 |
| `cublas_fp16`  |          **0.487993** |                **15.615773** |   **64.872353** |    10 |

`cublas_fp16` is fastest overall — FP16 weights halve GPU memory bandwidth per matmul, which is the binding bottleneck on the A100's HBM2e. `cublas` (FP32) is next. `flash` and `tiled` use custom CUDA matmul kernels without cuBLAS and are bandwidth-limited by the slower custom paths.

### Stage-Level Notes

The profiling CSVs are normalized to a common schema for `flash` and `cublas`, while `tiled` exposes the same conceptual stages with implementation-specific labels (`cpu_attn_ms` and `h2d_out_proj_res_ms`). The plots and summary table above map those labels back to the closest shared stage names so the comparison stays readable.

## � Debugging Notes & Lessons Learned

### `cublas_fp16` — Gibberish Output on 7B Model (Root Cause & Fix)

When `cublas_fp16` was first run against `llama2_7b.bin`, the binary produced gibberish at 58–66 tok/s. The root cause was a subtle pointer bug in weight loading.

**The `vocab_size` sign-encoding convention in llama2.c:**
The `.bin` checkpoint encodes whether the classifier matrix `wcls` is tied to the embedding table via the *sign* of `vocab_size`:
- `vocab_size > 0` → shared weights (`stories15M.bin`, `vocab_size = +32000`)
- `vocab_size < 0` → separate classifier (`llama2_7b.bin`, `vocab_size = -32000`)

**The bug:**
`build_transformer` reads the raw `vocab_size`, records `shared = (vocab_size > 0)`, then immediately calls `abs(vocab_size)`. By the time `map_weights_from_ptr` ran, `p->vocab_size` was always positive. The function re-derived `shared_weights` from this now-positive value and always set `w->wcls = token_embedding_table`. For `stories15M` (which is actually shared-weight) this was silently correct. For `llama2_7b` (separate classifier), the classifier pointer was set to the embedding table — producing valid FP16 arithmetic on the *wrong matrix* → coherent arithmetic, garbage logits, garbage output.

**The fix:**
`map_weights_from_ptr` now always sets `w->wcls = ptr` (the end-of-file classifier location). `upload_weights_to_gpu` was already correct: it checks `cw->shared_weights` (set by `build_transformer` *after* `map_weights_from_ptr` returns) and uses `d_token_embedding_table` instead when `shared_weights == 1`.

**Why stories15M was not affected:**
`stories15M.bin` has `vocab_size = +32000` (positive), so both the old and new code derived `shared_weights = 1` and used the embedding table — identical behaviour.

**Lesson:** When porting llama2.c weight-loading logic, the `abs(vocab_size)` step and the `wcls` pointer derivation must happen in the correct order. Re-deriving `shared_weights` after `abs()` silently corrupts the classifier pointer for every non-shared-weight model.

---

### FP16 VRAM Savings — Measured vs Estimated (llama2_7b on A100)

| Metric | FP32 (`cublas`) | FP16 (`cublas_fp16`) |
|--------|----------------|----------------------|
| Checkpoint on disk | 27.0 GB | 27.0 GB (loaded, then converted on CPU) |
| Weight VRAM on GPU | ~26 GB | ~12,602 MB (~12.3 GB) |
| VRAM saved | — | **~13.4 GB (~51%)** |
| Throughput (A100) | 31.06 tok/s | **66.15 tok/s** |
| Speedup vs FP32 cublas | — | **2.13×** |

The throughput gain comes from halved GPU memory bandwidth for weight reads. Loading half as many bytes per matrix-vector product approximately doubles the memory-bandwidth-bound throughput on the A100's HBM2e (2 TB/s).

---

### Auto-Download `--force` Fix

`modal volume get` exits with code 1 if the local destination already exists. The second run of `cublas_fp16` failed all three auto-download attempts because `profiling_cublas_fp16.txt` from the first run was still present. Fixed by adding `--force` to the `modal volume get` command in `modal_app.py`.

Manually retrieving an existing file:
```bash
modal volume get --force huggingface-cache profiling_cublas_fp16.txt .
```

---

### `--profile-enable` Partial Support for `cublas_fp16`

`llama2_cublas_fp16.cu` implements the `-P <pos>` flag (per-layer bottleneck analysis, printed to stdout), but does **not** implement the `-R <path>` flag (CSV output) used by the other three implementations. Passing `--profile-enable` with `--cuda-impl cublas_fp16` will print the 8-stage timing breakdown at the requested token position, but will not write a profiling CSV to the cache volume.

To trigger profiling for `cublas_fp16`:
```bash
modal run modal_app.py --cuda-impl cublas_fp16 --model llama2_7b.bin --profile-enable --profile-pos 10
```

---

## 💡 Usage Examples

### Complete CLI Reference

All commands start with `modal run modal_app.py`. Every argument below is optional and has a default.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `stories15M.bin` | Model file name. Small models use A10G, large (>1 GB) use A100. |
| `--prompt` | str | `"Once upon a time"` | Input text prompt. |
| `--steps` | int | `256` | Number of tokens to generate. |
| `--temperature` | float | `1.0` | Sampling temperature. `0.0` = greedy (deterministic). |
| `--topp` | float | `0.9` | Top-p nucleus sampling cutoff. |
| `--seed` | int | `42` | Random seed. |
| `--cuda-impl` | str | `cublas` | CUDA kernel variant: `cublas`, `flash`, `tiled`, `cublas_fp16`. |
| `--test-only` | flag | off | Only verify CUDA/GPU setup; skip inference. |
| `--list-models` | flag | off | Print all configured models and exit. |
| `--test-s3` | flag | off | Verify AWS credentials and S3 bucket connectivity, then exit. |
| `--use-boto3` | flag | off | Use `boto3` instead of the AWS CLI for S3 downloads. |
| `--cache-action` | str | — | Volume cache operation: `list`, `info`, or `clear`. |
| `--save-output` | flag | on | Save the full log to the cache volume as a `.txt` file. |
| `--output-file` | str | `profiling_<impl>.txt` | Override the output filename saved in the cache volume. |
| `--list-profiling` | flag | off | List all saved profiling files in the cache volume. |
| `--profile-enable` | flag | off | Enable per-layer CUDA event timing inside the binary. |
| `--profile-pos` | int | `10` | Token position at which per-layer timing is captured. |
| `--profile-csv` | str | `/cache/profiling_<impl>.csv` | Remote path for the stage-timing CSV. Use `/cache/` prefix for persistence. |
| `--download-dir` | str | `.` | Local directory where profiling artifacts are auto-downloaded after the run. |

`--cuda-impl` source file mapping:
- `cublas` → `llama2_cublas.cu` (binary: `llama2_cublas`)
- `flash` → `llama2_flash.cu` (binary: `llama2_flash`)
- `tiled` → `llama2_tiled.cu` (binary: `llama2_tiled`)
- `cublas_fp16` → `llama2_cublas_fp16.cu` (binary: `llama2_cublas_fp16`) — FP16 weights, FP32 accumulation

### Development Workflow

```bash
# 1. Test locally with small model
make run

# 2. Test CUDA locally without Modal auth
python modal_app.py --test-only

# 3. Test Modal setup
modal run modal_app.py --test-only

# 4. Test S3 connectivity  
modal run modal_app.py --test-s3

# 5. Run small model on Modal (cuBLAS implementation)
modal run modal_app.py --cuda-impl cublas --model stories15M.bin --prompt "Hello AI"

# 6. Run large model from S3 (cuBLAS implementation)
modal run modal_app.py --cuda-impl cublas --model llama2_7b.bin --prompt "Once upon a time"

# 7. Run the Flash implementation
modal run modal_app.py --cuda-impl flash --model llama2_7b.bin --prompt "Once upon a time"

# 8. Run the Flash implementation with profiling enabled
modal run modal_app.py --cuda-impl flash --model llama2_7b.bin --prompt "Once upon a time" --profile-enable --profile-pos 10

# 9. Run the tiled implementation
modal run modal_app.py --cuda-impl tiled --model llama2_7b.bin --prompt "Once upon a time"

# 10. Run the FP16 mixed-precision implementation (halves VRAM for weights)
modal run modal_app.py --cuda-impl cublas_fp16 --model stories15M.bin --prompt "Once upon a time"

# 11. Run FP16 on large model (saves ~13 GB VRAM vs FP32)
modal run modal_app.py --cuda-impl cublas_fp16 --model llama2_7b.bin --prompt "Once upon a time"

# 12. Run 30 profiling iterations for every implementation and archive the results
bash run_profiling_experiment.sh --impl all --runs 30 --model llama2_7b.bin
```

The Flash profiling command writes `profiling_flash.txt` and `profiling_flash.csv` to the cache volume and auto-downloads them locally after the run completes.
If you are running the shell script from Windows and Bash cannot find Modal, use `--modal-cmd modal.exe` or run the script from a shell where `modal` is on PATH.
The batch runner is a Bash script, so from PowerShell you should invoke it through `bash`; if Bash cannot see Python, add `--python-cmd py -3` or `--python-cmd python.exe`.

### Production Scenarios

```bash
# High-quality text generation (cuBLAS)
modal run modal_app.py \
  --cuda-impl cublas \
  --model llama2_7b.bin \
  --prompt "Write a short story about AI" \
  --steps 1024 \
  --temperature 0.7

# FP16 mixed-precision (saves ~13 GB VRAM, same quality)
modal run modal_app.py \
  --cuda-impl cublas_fp16 \
  --model llama2_7b.bin \
  --prompt "Write a short story about AI" \
  --steps 1024 \
  --temperature 0.7

# Fast prototyping (Flash)
modal run modal_app.py \
  --cuda-impl flash \
  --model stories15M.bin \
  --prompt "Quick test" \
  --steps 100 \
  --temperature 1.0

# Deterministic generation (cuBLAS)
modal run modal_app.py \
  --cuda-impl cublas \
  --model llama2_7b.bin \
  --prompt "Explain quantum computing" \
  --temperature 0.0 \
  --seed 42

# Tiled matmul comparison run
modal run modal_app.py \
  --cuda-impl tiled \
  --model llama2_7b.bin \
  --prompt "Compare tiled matmul and flash attention" \
  --steps 512 \
  --temperature 0.7

# Download profiling artifacts to a custom local folder
modal run modal_app.py \
  --cuda-impl cublas \
  --model llama2_7b.bin \
  --profile-enable \
  --profile-pos 10 \
  --download-dir ./results
```

### Cache Management

```bash
# Check cached models and profiling files
modal run modal_app.py --cache-action list

# Get cache storage statistics
modal run modal_app.py --cache-action info

# Clear cache to save space
modal run modal_app.py --cache-action clear

# List all saved profiling .txt / .csv files in the volume
modal run modal_app.py --list-profiling

# Manually download a specific file from the cache volume
modal volume get huggingface-cache profiling_cublas.txt
```

### Batch Profiling Runs

Use the shell runner to execute the same profiling experiment repeatedly and store each run under `profiling_runs/<impl>/run_##/`.

```bash
# Run 30 iterations for flash only
bash run_profiling_experiment.sh --impl flash --runs 30

# Windows-safe: force the Modal CLI executable explicitly
bash run_profiling_experiment.sh --impl flash --runs 30 --modal-cmd modal.exe

# Windows-safe: force the Python launcher explicitly for the stats step
bash run_profiling_experiment.sh --impl flash --runs 30 --modal-cmd modal.exe --python-cmd "py -3"

# Run all three implementations with 50 repetitions each
bash run_profiling_experiment.sh --impl all --runs 50 --model llama2_7b.bin
```

Each run directory contains the raw text log, the profiling CSV, and a `profile_stats.csv` summary.

### Profiling Methodology & Measurement Accuracy

The profiling system captures two different measurements:

#### 1. Per-Layer Profiling (Layer Index, Position-Triggered)
Captured at a specific token position (default: position 10) in layer 0:
- **RMSNorm (att)**: 0.018 ms
- **QKV Projections**: 0.731 ms
- **RoPE + KV Cache**: 0.028 ms
- **Flash Attention**: 0.034 ms
- **Output Proj + Residual**: 0.252 ms
- **RMSNorm (FFN)**: 0.009 ms
- **FFN W1+W3**: 0.731 ms
- **SwiGLU + W2**: 0.172 ms

**Extrapolation formula**: `1.975 ms/layer × 32 layers = 63.2 ms/token = 15.8 tok/s`

This is a **conservative snapshot** because:
- GPU cache is cold at early token positions
- Single kernel launch overhead per stage is not yet amortized
- Represents worst-case per-layer latency

#### 2. End-to-End Throughput (Full Generation)
Measured across all 256 generated tokens: **31.14 tok/s**

This is **2× higher** than the per-layer estimate because:
- **GPU warmup**: By token 50+, caches are hot and kernels launch faster
- **Kernel amortization**: Launch overhead spreads over many tokens
- **Pipeline smoothing**: Batch processing reduces per-token variance
- **Full model averaging**: Later layers and later tokens run faster than early position 10

#### Why the 2× Difference?

| Factor | Impact |
|--------|--------|
| **Cold GPU cache (pos=10)** | Layer 0 at position 10 is pessimistic |
| **CUDA event timing overhead** | Per-stage profiling adds synchronization cost |
| **GPU cache state** | Later tokens benefit from filled L2/L1 caches |
| **Kernel launch overhead** | Fixed cost amortized better over longer runs |
| **Per-layer snapshot bias** | Layer 0 ≠ average of all 32 layers |

#### Best Practices for Defensible Baselines

For IEEE-paper-quality profiling, use one of these approaches:

**Option 1: Skip Warmup (Steady-State Measurement)**
```bash
bash run_profiling_experiment.sh --impl flash --runs 30 --profile-pos 50
```
- Profile at position 50+ to skip cold-cache tokens
- Captures GPU at thermal equilibrium
- More representative of production inference

**Option 2: Full-Model End-to-End (Recommended)**
```bash
bash run_profiling_experiment.sh --impl flash --runs 30
```
- Measure entire generation pipeline
- Report both per-token average and per-layer breakdown
- Best for real-world throughput comparisons

**Option 3: Cold-Start Analysis (Variability Study)**
```bash
bash run_profiling_experiment.sh --impl flash --runs 30 --profile-pos 1 --profile-pos 10 --profile-pos 50
```
- Capture cold, warm, and hot GPU states
- Document the warmup curve
- Show 2–3× speedup from warmup effects

#### Statistical Validity

Run **at least 30 iterations** per implementation to:
- Average out warmup effects
- Capture GPU scheduling variance
- Compute robust confidence intervals (95% CI)
- Account for Modal cloud jitter

Output CSV files include:
- `n`: number of samples
- `mean`, `std`, `min`, `max`, `median`, `p95`: descriptive statistics
- `cv_pct`: coefficient of variation (shows variability)
- `ci95_low`, `ci95_high`: 95% confidence interval

### Getting Help

1. **Check Modal logs**: Modal provides detailed execution logs
2. **Verify S3 setup**: Use `--test-s3` command
3. **Test incrementally**: Start with small models, then scale up
4. **Monitor resources**: Check GPU memory usage and processing time

## 📝 Project Structure

```
llama2-cuda/
├── llama2.c                     # Original CPU-only implementation (karpathy/llama2.c)
├── llama2_cublas.cu             # cuBLAS-backed CUDA implementation
├── llama2_cublas_fp16.c         # CPU-only FP16 prototype (basis for cublas_fp16 variant)
├── llama2_cublas_fp16.cu        # FP16 weights / FP32 accumulation (cublas_fp16 variant)
├── llama2_flash.cu              # Flash attention CUDA implementation
├── llama2_tiled.cu              # Shared-memory tiled CUDA implementation
├── llama2_tiled_backup.cu       # Backup / experimental tiled implementation
├── modal_app.py                 # Modal deployment and cloud inference script
├── vllm_inference.py            # vLLM cloud inference via Modal
├── profile_stats.py             # Profiling statistics analysis and CSV summary
├── run_profiling_experiment.sh  # Batch profiling runner (Bash)
├── Makefile                     # Build configuration
├── win.c                        # Windows compat shim (karpathy/llama2.c)
├── win.h                        # Windows compat shim (karpathy/llama2.c)
├── tokenizer.bin                # Tokenizer binary (auto-downloaded by modal_app.py)
├── tokenizer.model              # SentencePiece tokenizer model (Meta Llama-2)
├── stories15M.bin               # Small test model (15M parameters)
├── docs/                        # Documentation assets and profiling plots
├── Tests/                       # Test scripts and debug CUDA implementations
├── profiling_runs/              # Per-run profiling artifacts (CSV + TXT)
└── README.md                    # This documentation



S3 Bucket Structure:
s3://llama2-model/
└── llama2_7b.bin          # Large model (26GB)
```


For additional support:
- [Modal Documentation](https://modal.com/docs)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
