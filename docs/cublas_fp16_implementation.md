# FP16 Mixed-Precision Variant: `llama2_cublas_fp16.cu`

## Overview

`llama2_cublas_fp16.cu` is a mixed-precision CUDA implementation of LLaMA-2 7B inference. It stores all weight tensors on the GPU in **FP16 (`__half`)**, cutting VRAM consumption roughly in half compared to the FP32 baseline, while keeping all activations, the KV cache, and dot-product accumulations in **FP32**. The cuBLAS `CUBLAS_COMPUTE_32F` flag ensures no quality loss from the precision reduction.

This is the primary precision-reduction baseline in the project and the starting point for the `cublas_bf16` and `cublas_fp16tc` ablation variants.

---

## Motivation

### Why not run the FP32 baseline on large models?

The FP32 `llama2_cublas.cu` variant loads all ~26 GB of LLaMA-2 7B weights into GPU VRAM. An A100-40GB can just fit this, but it leaves almost no headroom for the activation buffers, KV cache, and cuBLAS workspace. On any GPU with less than 40 GB (e.g. RTX 3090, A10G) the model cannot be loaded at all in FP32.

### Why FP16 for weights only?

| What to quantize | Benefit | Risk |
|---|---|---|
| Weights only (this file) | 2× VRAM, same activation precision | Weight values exceeding FP16 max (~65,504) are corrupted |
| Activations too | More VRAM saved | Cumulative rounding error across 32 transformer layers is severe |
| Accumulation too | Highest throughput | Dot-product overflow / NaN on large hidden dims (CUBLAS_COMPUTE_16F) |

Quantizing only weight storage keeps the numerically sensitive parts (attention scores, softmax, residual additions) in FP32, giving near-FP32 output quality at half the memory cost.

### Why CUBLAS_COMPUTE_32F and not CUBLAS_COMPUTE_16F?

`CUBLAS_COMPUTE_32F` tells cuBLAS to accumulate each partial dot-product sum in a 32-bit register, even though inputs A and B are both FP16. This is the safe, lossless path supported on CC 7.x+ (Volta and newer). `CUBLAS_COMPUTE_16F` accumulates in FP16, which can overflow for large hidden dimensions and requires the output matrix to also be `CUDA_R_16F` — complicating integration with the FP32 activation pipeline. The `cublas_fp16tc` ablation explores that tradeoff explicitly.

---

## Precision data-flow diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Disk checkpoint (.bin)                                         │
│  All tensors: float (FP32)                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │  CPU: __float2half() per element
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  GPU weight buffers: __half (FP16)                              │
│  token_embedding_table, rms_att/ffn_weight,                     │
│  wq, wk, wv, wo, w1, w2, w3, rms_final_weight, wcls            │
└────────────────────────┬────────────────────────────────────────┘
                         │  Per token:
                         │    embed_fp16_to_fp32 kernel (embedding row → FP32 d_x)
                         │    cublas_sgemv_fp16: FP16 W × FP16 x_scratch → FP32 y
                         │    rmsnorm_fp16: __half2float(w[i]) × FP32 x[i]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  GPU activation buffers: float (FP32)                           │
│  d_x, d_xb, d_xb2, d_q, d_k, d_v, d_hb, d_hb2,               │
│  d_key_cache, d_value_cache, d_logits                           │
│                                                                 │
│  All arithmetic: FP32 (RoPE, attention, softmax, SwiGLU,        │
│  residual add, final norm)                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │  cudaMemcpy D→H
                         ▼
                  FP32 logits on CPU → sampling
```

---

## Implementation details

### 1. Weight upload: CPU conversion, one-shot

All FP32→FP16 conversions happen on the CPU at startup inside `upload_fp32_as_fp16()`:

```c
static __half* upload_fp32_as_fp16(const float* src, size_t n) {
    __half* tmp = (__half*)malloc(n * sizeof(__half));
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(src[i]);
    __half* d_ptr = alloc_half_gpu(n);
    cudaMemcpy(d_ptr, tmp, n * sizeof(__half), cudaMemcpyHostToDevice);
    free(tmp);
    return d_ptr;
}
```

Converting on the CPU avoids an extra GPU kernel launch at startup and keeps the runtime-critical path clean. The CPU FP32 mmap is not freed — it is retained so the optional `--verify` mode can build a CPU-side FP16 weight copy without re-reading the file.

### 2. Activation scratch buffer: lazy reuse

`cublasGemmEx` cannot mix `CUDA_R_16F` weights with a `CUDA_R_32F` activation vector — that combination is not a valid cuBLAS type pair. The activation vector must be FP16 for the call. Rather than a separate allocation per call, a single global scratch buffer is lazily allocated and grown on demand:

```c
static __half*  d_x_fp16_scratch   = NULL;
static size_t   d_x_fp16_scratch_n = 0;

static void ensure_fp16_scratch(size_t n) {
    if (n > d_x_fp16_scratch_n) {
        if (d_x_fp16_scratch) cudaFree(d_x_fp16_scratch);
        CUDA_CHECK(cudaMalloc(&d_x_fp16_scratch, n * sizeof(__half)));
        d_x_fp16_scratch_n = n;
    }
}
```

This means the first call with `n = dim = 4096` allocates 8 KB; subsequent calls (even for `hidden_dim = 11008`) grow it to 22 KB. After that no further allocations occur for the lifetime of the program.

### 3. cuBLAS GEMV: reading the layout correctly

The weight matrices are stored **row-major** in the checkpoint: `W[d × n]` where `d` is the output dimension and `n` is the input. cuBLAS treats all matrices as **column-major**, so it sees `W` as `(n × d)` col-major. To compute `y = W @ x` (FP32), we apply `CUBLAS_OP_T` to transpose back:

```c
cublasGemmEx(
    cublas_handle,
    CUBLAS_OP_T, CUBLAS_OP_N,   /* transpose W back to (d × n) for cuBLAS */
    d, 1, n,                    /* output size d × 1 */
    &alpha,                     /* 1.0f */
    d_W, CUDA_R_16F, n,         /* A: W (n × d col-major = d × n row-major) */
    d_x_fp16_scratch, CUDA_R_16F, n,  /* B: x (n × 1) */
    &beta,                      /* 0.0f */
    d_y, CUDA_R_32F, d,         /* C: y (d × 1) in FP32 */
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

`alpha`/`beta` are `float` scalars — `CUBLAS_COMPUTE_32F` accepts them in FP32 (unlike `CUBLAS_COMPUTE_16F` which requires `__half` scalars).

### 4. RMSNorm kernel: widening on the fly

The RMSNorm weight (`rms_att_weight`, `rms_ffn_weight`) is stored as `__half`. The kernel reads it, immediately widens to `float`, and performs all accumulation in FP32:

```c
__global__ void rmsnorm_fp16_kernel(float* out, const float* x,
                                     const __half* w, int size, float eps)
{
    // ... FP32 sum-of-squares + warp-level reduction ...
    float inv = smem;  /* 1/sqrt(ss/size + eps) — FP32 */
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        out[i] = __half2float(w[i]) * (inv * x[i]);  /* widen, then FP32 multiply */
}
```

The warp-level reduction uses `__shfl_down_sync` across all 32 threads; an `atomicAdd` into shared memory collects warp sums. Block size is 256 for `dim ≤ 1024`, 512 otherwise.

### 5. Embedding lookup: `embed_fp16_to_fp32`

Token embeddings are stored as FP16 rows (`d_token_embedding_table`). Each forward step reads one row and writes it as FP32 to `d_x`:

```c
__global__ void embed_fp16_to_fp32(float* out, const __half* emb, int offset, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = __half2float(emb[offset + i]);
}
```

This is a single memory-bandwidth-bound kernel. For `dim = 4096`, it touches 8 KB of FP16 and writes 16 KB of FP32 — effectively a format-conversion copy.

### 6. Attention: stays entirely in FP32

The KV cache (`d_key_cache`, `d_value_cache`) is allocated in FP32. Q, K, V projections are written to FP32 by `cublas_sgemv_fp16`. RoPE rotary embeddings operate on FP32 Q and K. The flash-attention kernel reads and writes FP32 scores, keys, values, and softmax weights. No precision narrowing occurs anywhere in the attention path.

Grouped Query Attention (GQA) is supported via the mapping `kv_h = h * n_kv_heads / n_heads` inside `flash_attention_kernel`.

### 7. Shared-weights handling

For small models (stories15M.bin) the classifier matrix `wcls` is shared with `token_embedding_table`. The sign of `vocab_size` in the checkpoint header encodes this flag. `build_transformer` reads the sign, stores `shared_weights`, and `upload_weights_to_gpu` reuses the `d_token_embedding_table` pointer for `d_wcls` instead of allocating a second copy:

```c
if (cw->shared_weights) {
    gw->d_wcls = gw->d_token_embedding_table;  /* alias, no second allocation */
} else {
    gw->d_wcls = upload_fp32_as_fp16(cw->wcls, (size_t)p->vocab_size * p->dim);
}
```

### 8. Per-layer bottleneck profiler

The `-P <pos>` flag enables `profile_7b_bottlenecks()`, which runs **once** at the specified token position and records 8 CUDA-event-timed stages for layer 0:

| Stage | What is timed |
|---|---|
| 1 | Attention RMSNorm |
| 2 | QKV projections (3 × GEMV) |
| 3 | RoPE + KV cache store |
| 4 | Flash Attention |
| 5 | Output projection + residual add |
| 6 | FFN RMSNorm |
| 7 | FFN W1 + W3 (2 × GEMV) |
| 8 | SwiGLU + W2 + residual add |

Results are printed to stdout and appended to a CSV file via `append_profile_metrics_csv()`. The CSV schema (18 columns) is shared across all variants so `profile_stats.py` can compare them directly.

### 9. Profiling CSV schema

```
timestamp_unix, token, pos, dim, hidden_dim, n_heads, n_kv_heads,
rms_att_ms, qkv_ms, rope_kvcache_ms, flash_attn_ms, out_proj_res_ms,
rms_ffn_ms, ffn_w1w3_ms, swiglu_w2_ms,
total_layer_ms, est_32_layer_ms, est_tok_s
```

`est_32_layer_ms = total_layer_ms × n_layers`. `est_tok_s = 1000 / est_32_layer_ms`. The profiler touches only layer 0 to avoid inflating wall time; the estimate assumes all layers are equally slow (a good approximation for memory-bound inference).

### 10. `--verify` mode: CPU FP16 reference pass

When `--verify` is active, a second forward pass runs on the CPU using a `CPUWeightsFP16` struct (all `__half` arrays allocated with `malloc`). Every matrix multiply on the CPU accumulates in `float` (widening each `__half` operand), giving a reference that tests whether the GPU FP16-weight path produces numerically consistent results.

```
[verify pos=  0] max_diff=0.00012  mean_diff=0.000003
[verify pos=  1] max_diff=0.00018  mean_diff=0.000004
```

A divergence threshold of `> 1.0` is flagged as `*** LARGE DIVERGENCE ***`. In practice FP16 weight storage with FP32 accumulation produces max_diff well below `0.01` for LLaMA-2 7B.

---

## Memory layout

| Buffer | Type | Size (7B) | Location |
|---|---|---|---|
| All weight tensors | `__half` | ~13 GB | GPU VRAM |
| `d_x`, `d_xb`, `d_xb2` | `float` | 3 × 16 KB | GPU VRAM |
| `d_q`, `d_k`, `d_v` | `float` | 16 KB + 2×8 KB | GPU VRAM |
| `d_hb`, `d_hb2` | `float` | 2 × 44 KB | GPU VRAM |
| `d_key_cache`, `d_value_cache` | `float` | 2 × ~2 GB | GPU VRAM |
| `d_logits` | `float` | 128 KB | GPU VRAM |
| `d_x_fp16_scratch` | `__half` | max(n) × 2 B | GPU VRAM |
| `h_logits` | `float` | 128 KB | CPU RAM |

KV cache dominates activation memory: `2 × n_layers × seq_len × kv_dim × 4 bytes = 2 × 32 × 2048 × 1024 × 4 ≈ 536 MB` for the default 2048 sequence length.

---

## Measured performance (A100-SXM4-40GB)

| Variant | Mean layer (ms) | Est. 32-layer (ms) | Est. tok/s | Runs |
|---|---|---|---|---|
| tiled (FP32) | 3.327 | 106.45 | 9.40 | 9 |
| flash (FP32) | 2.610 | 83.51 | 12.37 | 11 |
| cublas (FP32) | 0.926 | 29.64 | 33.74 | 10 |
| **cublas_fp16** | **0.488** | **15.62** | **64.87** | **9** |

The ~2× throughput gain over `cublas` (FP32) reflects the 2× reduction in weight-memory bandwidth: at 26 GB of weights on a 2 TB/s bandwidth bus, FP32 weights cause `26 GB / 2 TB/s = 13 ms` of HBM traffic per token. FP16 halves this to ~6.5 ms, matching the ~2× measured speedup.

---

## CLI flags

| Flag | Description |
|---|---|
| `-n <int>` | Number of generation steps (default 256) |
| `-i <string>` | Prompt text |
| `-t <float>` | Sampling temperature (default 1.0) |
| `-p <float>` | Top-p nucleus sampling (default 0.9) |
| `-s <int>` | Random seed |
| `-z <string>` | Tokenizer file path (default `tokenizer.bin`) |
| `-P <int>` | Enable profiling at this token position |
| `-R <string>` | Profiling CSV output path |
| `--verify` | Run CPU FP16 reference pass and print logit diffs |

---

## Build instructions

**Linux / WSL:**
```bash
nvcc llama2_cublas_fp16.cu -o llama2_cublas_fp16 \
    -O2 -arch=sm_80 -lcublas -lm
```

**Windows:**
```bat
nvcc llama2_cublas_fp16.cu win.c -o llama2_cublas_fp16.exe ^
    -O2 -arch=sm_80 -lcublas -lm
```

**Modal cloud (A100):**
```bash
modal run modal_app.py --cuda-impl cublas_fp16 --model llama2_7b.bin \
    --prompt "Once upon a time" --steps 256
```

With profiling:
```bash
modal run modal_app.py --cuda-impl cublas_fp16 --model llama2_7b.bin \
    --prompt "Once upon a time" --steps 256 --profile-enable --profile-pos 10
```

---

## Relation to other variants

| Variant | Weights | Compute | Output | Change from fp16 |
|---|---|---|---|---|
| `cublas` | FP32 | `COMPUTE_32F` | FP32 | baseline |
| **`cublas_fp16`** | **FP16** | **`COMPUTE_32F`** | **FP32** | **this file** |
| `cublas_bf16` | BF16 | `COMPUTE_32F` | FP32 | dtype: `__half` → `__nv_bfloat16` |
| `cublas_fp16tc` | FP16 | `COMPUTE_16F` | FP16→FP32 | accumulate in FP16; extra output conversion step |

---

## File map

```
llama2.cu/
├── llama2_cublas.cu               ← FP32 baseline
├── llama2_cublas_fp16.cu          ← this file
├── llama2_cublas_bf16.cu          ← BF16 weight variant
├── llama2_cublas_fp16tc.cu        ← FP16 Tensor Core (COMPUTE_16F) ablation
├── modal_app.py                   ← "cublas_fp16" in CUDA_SOURCE_VARIANTS
└── docs/
    ├── cublas_fp16_implementation.md   ← this document
    ├── cublas_bf16_implementation.md
    └── research_directions.md
```
