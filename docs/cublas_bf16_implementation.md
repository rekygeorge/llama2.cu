# BF16 Precision Variant: `llama2_cublas_bf16.cu`

## Overview

`llama2_cublas_bf16.cu` is a new mixed-precision CUDA implementation of LLaMA-2 7B inference that stores all weight tensors in **BF16 (Brain Float 16)** on the GPU, while keeping activations in FP32. It is derived directly from `llama2_cublas_fp16.cu` by performing a systematic substitution of the FP16 types, intrinsics, and cuBLAS data-type flags with their BF16 equivalents.

The file is a self-contained drop-in that accepts the same CLI flags as all other variants in this project and integrates with `modal_app.py` via the `--cuda-impl cublas_bf16` option.

---

## Why BF16?

### Floating-point format comparison

| Format | Sign | Exponent | Mantissa | Total bits | Exponent range |
|--------|------|----------|----------|------------|----------------|
| FP32   | 1    | 8        | 23       | 32         | ±3.4 × 10³⁸   |
| FP16   | 1    | 5        | 10       | 16         | ±6.5 × 10⁴    |
| BF16   | 1    | 8        | 7        | 16         | ±3.4 × 10³⁸   |

BF16 occupies the same 16-bit storage as FP16 but trades mantissa precision for the full FP32 exponent range. The consequences for LLM inference are significant:

**FP16 problem — weight overflow:**  
LLaMA-2 7B weight matrices (e.g., `wq`, `wk`, `w1`) contain values that can exceed the FP16 max of ~65,504. When a weight is silently clamped to `inf` during the FP32→FP16 conversion, the GEMM output for that row is corrupted, producing `inf` or `NaN` in the logits and nonsense tokens.

**BF16 solution — same dynamic range as FP32:**  
Truncating 23 mantissa bits to 7 rounds the stored value slightly but never overflows. A FP32 weight of value `v` is stored as `round_to_nearest(v)` where the rounding error is at most $2^{-7}$ of the value's magnitude (≈0.8%), versus ≈0.1% for FP16. In practice the 7-bit mantissa is sufficient for transformer weight matrices: attention heads average over many terms, so individual rounding errors cancel out.

**Memory bandwidth — identical to FP16:**  
Both formats require 16 bits per element. The VRAM footprint and HBM bandwidth consumed during a weight-matrix GEMV are identical. Any throughput difference between the two variants is caused only by cuBLAS kernel selection, not memory traffic.

---

## Design decisions

### 1. Weight storage: BF16; activations: FP32

The same mixed-precision strategy used in `llama2_cublas_fp16.cu` is carried over unchanged:

```
FP32 checkpoint on disk
        ↓  CPU: convert on load
BF16 weights on GPU (halves VRAM vs FP32)
        ↓  cuBLAS GemmEx: BF16 × BF16 → FP32 accumulation (CUBLAS_COMPUTE_32F)
FP32 activations: x, xb, q, k, v, hb, hb2, key_cache, value_cache, logits
```

Keeping activations in FP32 avoids numerical drift across layers. The GEMM accumulates in FP32 internally (cuBLAS handles this via `CUBLAS_COMPUTE_32F`), so the BF16 precision loss only affects how each weight element is stored, not how dot products accumulate.

### 2. `CUDA_R_16BF` instead of `CUDA_R_16F`

The only change inside the cuBLAS call is the data-type tag passed for the A and B matrices:

```c
// FP16 version:
CUBLAS_CHECK(cublasGemmEx(
    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    d, 1, n, &alpha,
    d_W,              CUDA_R_16F, n,   /* weight: FP16 */
    d_x_fp16_scratch, CUDA_R_16F, n,   /* activation: FP16 */
    &beta,
    d_y,              CUDA_R_32F, d,   /* output: FP32 */
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

// BF16 version:
CUBLAS_CHECK(cublasGemmEx(
    cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
    d, 1, n, &alpha,
    d_W,              CUDA_R_16BF, n,  /* weight: BF16 */
    d_x_bf16_scratch, CUDA_R_16BF, n,  /* activation: BF16 */
    &beta,
    d_y,              CUDA_R_32F,  d,  /* output: FP32 */
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
```

`CUDA_R_16BF` was introduced in CUDA 11.0 and is natively supported by the A100 (SM80) Tensor Cores. On older GPUs cuBLAS will fall back to a software emulation path that is slower than native FP16.

### 3. On-the-fly activation conversion

Like the FP16 variant, the activation vector is stored as FP32 and must be type-cast before each GEMV. A small scratch buffer (`d_x_bf16_scratch`) is kept alive for the lifetime of the program and resized lazily:

```c
// fp32_to_bf16_kernel — one thread per element
__global__ void fp32_to_bf16_kernel(__nv_bfloat16* out, const float* in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}
```

`__float2bfloat16_rn` rounds to nearest, which is the standard rounding mode for training. Using round-to-nearest (rather than round-toward-zero, i.e., simple truncation) minimises systematic bias when the activation values are small.

### 4. RMSNorm kernel adapted for BF16 weights

The RMSNorm kernel reads the weight vector in BF16 and immediately widens to FP32 for the normalization arithmetic:

```c
__global__ void rmsnorm_bf16_kernel(float* out, const float* x,
                                     const __nv_bfloat16* w, int size, float eps)
{
    // ... sum-of-squares reduction in FP32 ...
    float inv = smem;   /* 1/sqrt(ss/size + eps) in FP32 */
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        out[i] = __bfloat162float(w[i]) * (inv * x[i]);
}
```

The widening `__bfloat162float(w[i])` is a zero-cost reinterpret on SM80 (the hardware sign-extends the 8-bit exponent and zero-pads the mantissa to 23 bits). The multiply and accumulate then proceed in FP32.

### 5. Embedding lookup uses `embed_bf16_to_fp32`

Token embeddings are stored as BF16 rows in `d_token_embedding_table`. Each forward pass reads one row and copies it to the FP32 activation buffer `d_x`:

```c
__global__ void embed_bf16_to_fp32(float* out, const __nv_bfloat16* emb,
                                    int offset, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = __bfloat162float(emb[offset + i]);
}
```

### 6. CPU conversion on load, not at runtime

All FP32→BF16 conversions happen once during `build_transformer` on the CPU, before uploading to the GPU. This avoids a GPU-side conversion kernel at startup and keeps the runtime path identical to the FP16 variant.

```c
static __nv_bfloat16* upload_fp32_as_bf16(const float* src, size_t n) {
    __nv_bfloat16* tmp = (__nv_bfloat16*)malloc(n * sizeof(__nv_bfloat16));
    for (size_t i = 0; i < n; i++) tmp[i] = __float2bfloat16_rn(src[i]);
    __nv_bfloat16* d_ptr = alloc_bf16_gpu(n);
    cudaMemcpy(d_ptr, tmp, n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    free(tmp);
    return d_ptr;
}
```

### 7. SM80 compatibility check at runtime

BF16 Tensor Core acceleration requires SM80 (A100) or newer. The `main()` function emits a non-fatal warning on older devices so the program still runs (using a slower software path) rather than failing silently:

```c
if (prop.major < 8) {
    fprintf(stderr,
        "WARNING: BF16 Tensor Core support requires SM80+ (A100).\n"
        "         This GPU (SM%d%d) will use BF16 in software.\n",
        prop.major, prop.minor);
}
```

### 8. Profiling: identical CSV schema

The `profile_7b_bottlenecks` function and `append_profile_metrics_csv` are carried over verbatim. The 18-column CSV schema is identical to all other variants:

```
timestamp_unix, token, pos, dim, hidden_dim, n_heads, n_kv_heads,
rms_att_ms, qkv_ms, rope_kvcache_ms, flash_attn_ms, out_proj_res_ms,
rms_ffn_ms, ffn_w1w3_ms, swiglu_w2_ms, total_layer_ms, est_32_layer_ms, est_tok_s
```

This means `profile_stats.py` can compute statistics over `cublas_bf16` runs without any schema changes.

---

## Complete substitution table

| `llama2_cublas_fp16.cu` | `llama2_cublas_bf16.cu` | Reasoning |
|-------------------------|-------------------------|-----------|
| `#include <cuda_fp16.h>` | `#include <cuda_bf16.h>` | BF16 type definitions and intrinsics |
| `__half` | `__nv_bfloat16` | BF16 scalar type (CUDA ≥ 11.0) |
| `__float2half(x)` | `__float2bfloat16_rn(x)` | Round-to-nearest conversion |
| `__half2float(x)` | `__bfloat162float(x)` | Widen to FP32 for arithmetic |
| `CUDA_R_16F` | `CUDA_R_16BF` | cuBLAS data-type enum for BF16 |
| `GPUWeightsFP16` | `GPUWeightsBF16` | Struct name updated for clarity |
| `CPUWeightsFP16` | `CPUWeightsBF16` | CPU verify struct |
| `fp32_to_fp16_kernel` | `fp32_to_bf16_kernel` | Activation scratch conversion kernel |
| `d_x_fp16_scratch` | `d_x_bf16_scratch` | Scratch buffer pointer |
| `ensure_fp16_scratch` | `ensure_bf16_scratch` | Lazy allocator |
| `cublas_sgemv_fp16` | `cublas_sgemv_bf16` | GEMV wrapper |
| `rmsnorm_fp16_kernel` | `rmsnorm_bf16_kernel` | RMSNorm kernel |
| `cuda_rmsnorm_fp16` | `cuda_rmsnorm_bf16` | RMSNorm launcher |
| `embed_fp16_to_fp32` | `embed_bf16_to_fp32` | Embedding lookup kernel |
| `upload_fp32_as_fp16` | `upload_fp32_as_bf16` | CPU→GPU upload helper |
| `alloc_half_gpu` | `alloc_bf16_gpu` | GPU allocator |
| `cpu_fp16_to_float` | `cpu_bf16_to_float` | CPU verify scalar cast |
| `cpu_rmsnorm_fp16` | `cpu_rmsnorm_bf16` | CPU verify RMSNorm |
| `cpu_matmul_fp16` | `cpu_matmul_bf16` | CPU verify matmul |
| `forward_cpu_fp16` | `forward_cpu_bf16` | CPU verify forward pass |
| `build_cpu_fp16_weights` | `build_cpu_bf16_weights` | CPU verify weight builder |
| Default CSV filename | `cublas_bf16_profile_metrics.csv` | Distinguishable artifact name |

Everything not listed above — `rope_kernel`, `swiglu_kernel`, `add_kernel`, `flash_attention_kernel`, the tokenizer, sampler, generation loop, and CLI argument parsing — is **unchanged** because those components operate entirely on FP32 activations and are precision-format agnostic.

---

## Integration with `modal_app.py`

The `CUDA_SOURCE_VARIANTS` dictionary in `modal_app.py` was extended:

```python
"cublas_bf16": {
    "source_file": "llama2_cublas_bf16.cu",
    "binary_name": "llama2_cublas_bf16",
},
```

This enables all four standard workflows:

```bash
# Basic inference
modal run modal_app.py --cuda-impl cublas_bf16 --model llama2_7b.bin \
    --prompt "Once upon a time" --steps 256

# Profiling with CSV output
modal run modal_app.py --cuda-impl cublas_bf16 --model llama2_7b.bin \
    --prompt "Once upon a time" --steps 256 \
    --profile-enable --profile-pos 50

# Verify GPU output matches CPU BF16 reference (--verify flag passed through run_cmd)
modal run modal_app.py --cuda-impl cublas_bf16 --model llama2_7b.bin \
    --prompt "Once upon a time" --steps 16
```

Profiling artifacts are saved to:
- **Container:** `/cache/profiling_runs/cublas_bf16/profiling_cublas_bf16.csv`
- **Local:** `profiling_runs/cublas_bf16/profiling_cublas_bf16.csv`

---

## Build instructions (local)

**Linux / WSL:**
```bash
nvcc llama2_cublas_bf16.cu -o llama2_cublas_bf16 \
    -O3 -arch=sm_80 -std=c++17 \
    -lcublas -lcudart

./llama2_cublas_bf16 llama2_7b.bin \
    -z tokenizer.bin -i "Once upon a time" -n 256
```

**Windows (with `win.c`):**
```bat
nvcc llama2_cublas_bf16.cu win.c -o llama2_cublas_bf16.exe ^
    -O3 -arch=sm_80 -std=c++17 ^
    -lcublas -lcudart

llama2_cublas_bf16.exe llama2_7b.bin ^
    -z tokenizer.bin -i "Once upon a time" -n 256
```

> `-arch=sm_80` is required for native BF16 Tensor Core kernels. Lower compute capabilities will still compile but will run on a slower emulation path.

---

## Expected performance vs. FP16

Both BF16 and FP16 variants use identical VRAM (16 bits/weight) and identical cuBLAS GEMV patterns. On A100 (SM80):

- **GEMV throughput** — cuBLAS selects a Tensor-Core-backed routine for both `CUDA_R_16F` and `CUDA_R_16BF` with `CUBLAS_COMPUTE_32F`. Actual kernel selection may differ slightly, leading to small (<5%) throughput differences between variants that vary run-to-run.
- **Expected tok/s** — approximately the same as `cublas_fp16` (≈64–65 tok/s on A100 for 7B), since the compute and memory patterns are identical.
- **Numerical quality** — BF16 is expected to produce outputs closer to FP32 because it eliminates weight-overflow artifacts. The `--verify` flag will show smaller `max_diff` values for BF16 vs FP16 for any model whose weights exceed ~1,000.

---

## File map

```
llama2.cu/
├── llama2_cublas_fp16.cu          ← original FP16 variant (reference)
├── llama2_cublas_bf16.cu          ← this file (BF16 variant)
├── modal_app.py                   ← "cublas_bf16" added to CUDA_SOURCE_VARIANTS
└── docs/
    └── cublas_bf16_implementation.md   ← this document
```
