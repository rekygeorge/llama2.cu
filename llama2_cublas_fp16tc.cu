/* llama2_cublas_fp16tc.cu — FP16 Tensor Core (full FP16 compute) Llama-2 inference
 *
 * Precision ablation variant of llama2_cublas_fp16.cu:
 *
 *   cublas_fp16   (baseline): CUBLAS_COMPUTE_32F — weights FP16, accumulate FP32 (safe)
 *   cublas_fp16tc (this file): CUBLAS_COMPUTE_16F — weights FP16, accumulate FP16 (fast, lossy)
 *
 * The single change is in cublasGemmEx:
 *   - CUBLAS_COMPUTE_32F  →  CUBLAS_COMPUTE_16F
 *   - alpha/beta scalars:  float  →  __half  (required by cuBLAS when compute type is FP16)
 *
 * Why do this?
 *   CUBLAS_COMPUTE_16F lets cuBLAS select Tensor Core kernels that accumulate in FP16
 *   rather than widening each partial sum to FP32.  On SM80 (A100) this can increase
 *   effective TFLOP/s by using more of the FP16 peak bandwidth, at the cost of reduced
 *   numerical precision in the dot products.
 *
 * Trade-off:
 *   - Faster:  fewer register transfers; FP16 accumulation pipeline has higher throughput.
 *   - Lossy:   for large hidden dims (e.g. dim=4096), summing 4096 FP16 products in FP16
 *              can lose ~1-2 ULP per step; the effect is visible as slightly higher
 *              logit deviation vs the FP32-accumulation baseline.
 *   - Stable:  in practice LLaMA-2 7B weights are well-conditioned; output text quality
 *              is usually indistinguishable, but divergence from FP32 is measurable.
 *
 * Weights:     __half (FP16) on GPU  →  halves VRAM vs FP32
 * Arithmetic:  CUBLAS_COMPUTE_16F   →  FP16 accumulation (fast but lossy)
 * Activations: float  (FP32) on GPU
 *
 * Build (Linux/WSL):
 *   nvcc llama2_cublas_fp16tc.cu -o llama2_cublas_fp16tc -lcublas -lm -O2
 * Build (Windows):
 *   nvcc llama2_cublas_fp16tc.cu win.c -o llama2_cublas_fp16tc -lcublas -lm -O2
 * Usage (same CLI as run.c):
 *   ./llama2_cublas_fp16tc model.bin -n 256 -i "Once upon a time" [--verify]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include <fcntl.h>
#include <stdbool.h>

#ifdef _WIN32
  #include "win.h"
#else
  #include <unistd.h>
  #include <sys/mman.h>
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// ----------------------------------------------------------------------------
// Error checking macros

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static cublasHandle_t cublas_handle;

/* Profiling globals (triggered by -P <pos> flag) */
static bool g_enable_profile    = false;
static bool g_profile_triggered = false;
static int  g_profile_pos       = 10;
static char g_profile_csv_path[512] = "cublas_fp16tc_profile_metrics.csv";
#ifdef DUMP_LOGITS
static char g_logit_bin_path[512] = "";  /* -L flag: full path for raw float32 logit dump */
static char g_token_ids_path[512] = "";  /* -T flag: full path for generated token IDs JSON */
#endif

static void append_profile_metrics_csv(
    int token, int pos, int dim, int hidden_dim, int n_heads, int n_kv_heads,
    float stage1_ms, float stage2_ms, float stage3_ms, float stage4_ms,
    float stage5_ms, float stage6_ms, float stage7_ms, float stage8_ms,
    float total_layer_ms, float est_32_layer_ms, float est_tok_s
) {
    FILE* csv_file = fopen(g_profile_csv_path, "a");
    if (!csv_file) {
        fprintf(stderr, "Error: could not open %s for writing\n", g_profile_csv_path);
        return;
    }

    fseek(csv_file, 0, SEEK_END);
    long file_size = ftell(csv_file);

    if (file_size == 0) {
        fprintf(csv_file,
                "timestamp_unix,token,pos,dim,hidden_dim,n_heads,n_kv_heads,"
                "rms_att_ms,qkv_ms,rope_kvcache_ms,flash_attn_ms,out_proj_res_ms,"
                "rms_ffn_ms,ffn_w1w3_ms,swiglu_w2_ms,total_layer_ms,est_32_layer_ms,est_tok_s\n");
    }

    fprintf(csv_file,
            "%lld,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (long long)time(NULL),
            token, pos, dim, hidden_dim, n_heads, n_kv_heads,
            stage1_ms, stage2_ms, stage3_ms, stage4_ms,
            stage5_ms, stage6_ms, stage7_ms, stage8_ms,
            total_layer_ms, est_32_layer_ms, est_tok_s);
    fclose(csv_file);
}

// ----------------------------------------------------------------------------
// Config / model structures

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

/* CPU-side: FP32 mmap'd pointers (temporary, freed after GPU upload) */
typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
    int    shared_weights; /* 1 if wcls == token_embedding_table */
} CPUWeightsFP32;

/* GPU-side: FP16 weight tensors */
typedef struct {
    __half* d_token_embedding_table; /* kept FP16; embedding lookup converts row */
    __half* d_rms_att_weight;
    __half* d_rms_ffn_weight;
    __half* d_wq;
    __half* d_wk;
    __half* d_wv;
    __half* d_wo;
    __half* d_w1;
    __half* d_w2;
    __half* d_w3;
    __half* d_rms_final_weight;
    __half* d_wcls;               /* may be same allocation as d_token_embedding_table */
    int     shared_weights;
} GPUWeightsFP16;

/* All activation buffers on GPU are FP32 */
typedef struct {
    float* d_x;
    float* d_xb;
    float* d_xb2;
    float* d_hb;
    float* d_hb2;
    float* d_q;
    float* d_k;
    float* d_v;
    float* d_key_cache;
    float* d_value_cache;
    float* d_logits;
    /* host-side logits for sampling */
    float* h_logits;
} RunState;

typedef struct {
    Config         config;
    CPUWeightsFP32 cpu_w;
    GPUWeightsFP16 gpu_w;
    RunState       state;
    int            fd;
    float*         data;     /* mmap base */
    ssize_t        file_size;
} Transformer;

// ----------------------------------------------------------------------------
// CPU forward pass state (used only when --verify is active)

typedef struct {
    /* __half on CPU (cuda_fp16.h defines __half for host code too) */
    __half* token_embedding_table;
    __half* rms_att_weight;
    __half* rms_ffn_weight;
    __half* wq;
    __half* wk;
    __half* wv;
    __half* wo;
    __half* w1;
    __half* w2;
    __half* w3;
    __half* rms_final_weight;
    __half* wcls;
    int     shared_weights;
} CPUWeightsFP16;

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* key_cache;
    float* value_cache;
    float* att;
    float* logits;
} CPURunState;

// ----------------------------------------------------------------------------
// GPU kernels

/* FP16 RMSNorm: read FP16 weight, compute in FP32 */
__global__ void rmsnorm_fp16_kernel(float* __restrict__ out,
                                     const float* __restrict__ x,
                                     const __half* __restrict__ w,
                                     int size, float eps)
{
    __shared__ float smem;
    if (threadIdx.x == 0) smem = 0.0f;
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        ss += x[i] * x[i];
    /* block reduction */
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (threadIdx.x % warpSize == 0)
        atomicAdd(&smem, ss);
    __syncthreads();
    if (threadIdx.x == 0) smem = 1.0f / sqrtf(smem / size + eps);
    __syncthreads();
    float inv = smem;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        out[i] = __half2float(w[i]) * (inv * x[i]);
}

static void cuda_rmsnorm_fp16(float* d_out, const float* d_x,
                               const __half* d_w, int size)
{
    int block = (size < 1024) ? 256 : 512;
    rmsnorm_fp16_kernel<<<1, block>>>(d_out, d_x, d_w, size, 1e-5f);
}

/* Copy a single FP16 embedding row to FP32 activation buffer */
__global__ void embed_fp16_to_fp32(float* __restrict__ out,
                                    const __half* __restrict__ emb,
                                    int offset, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = __half2float(emb[offset + i]);
}

/* RoPE rotary embedding kernel (operates on FP32 Q/K) */
__global__ void rope_kernel(float* __restrict__ q,
                             float* __restrict__ k,
                             int n_heads, int n_kv_heads,
                             int head_size, int pos)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head = head_size / 2;
    int total_q = n_heads * half_head;
    int total_k = n_kv_heads * half_head;

    if (idx < total_q) {
        int h = idx / half_head;
        int i = (idx % half_head) * 2;
        float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
        float val  = pos * freq;
        float fcr  = cosf(val), fci = sinf(val);
        float* ptr = q + h * head_size;
        float q0 = ptr[i], q1 = ptr[i+1];
        ptr[i]   = q0 * fcr - q1 * fci;
        ptr[i+1] = q0 * fci + q1 * fcr;
    }
    if (idx < total_k) {
        int h = idx / half_head;
        int i = (idx % half_head) * 2;
        float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
        float val  = pos * freq;
        float fcr  = cosf(val), fci = sinf(val);
        float* ptr = k + h * head_size;
        float k0 = ptr[i], k1 = ptr[i+1];
        ptr[i]   = k0 * fcr - k1 * fci;
        ptr[i+1] = k0 * fci + k1 * fcr;
    }
}

/* SwiGLU: hb = silu(hb) * hb2, in-place */
__global__ void swiglu_kernel(float* __restrict__ hb,
                               const float* __restrict__ hb2,
                               int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = hb[i];
        v = v / (1.0f + expf(-v));   /* silu */
        hb[i] = v * hb2[i];
    }
}

/* Residual add: a += b */
__global__ void add_kernel(float* __restrict__ a,
                            const float* __restrict__ b,
                            int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) a[i] += b[i];
}

/* Flash Attention (multi-head, causal) — one CUDA block per head.
 * This is a minimal tiled implementation; the kv-cache is already in FP32. */
__global__ void flash_attention_kernel(const float* __restrict__ q,
                                        const float* __restrict__ key_cache,
                                        const float* __restrict__ val_cache,
                                        float* __restrict__ out,
                                        int seq_len, int head_size,
                                        int n_heads, int n_kv_heads,
                                        float scale)
{
    int h = blockIdx.x;            /* one block per query head */
    int kv_h = h * n_kv_heads / n_heads; /* GQA mapping */

    extern __shared__ float smem[];
    float* scores = smem;          /* [seq_len] scores for this head */

    const float* qh = q + h * head_size;
    float* outh = out + h * head_size;

    /* dot products */
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const float* kh = key_cache + t * n_kv_heads * head_size + kv_h * head_size;
        float s = 0.0f;
        for (int d = 0; d < head_size; d++) s += qh[d] * kh[d];
        scores[t] = s * scale;
    }
    __syncthreads();

    /* softmax (numerically stable) — single thread for simplicity */
    if (threadIdx.x == 0) {
        float mx = -1e30f;
        for (int t = 0; t < seq_len; t++) if (scores[t] > mx) mx = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < seq_len; t++) scores[t] *= inv_sum;
    }
    __syncthreads();

    /* weighted sum of values */
    for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const float* vh = val_cache + t * n_kv_heads * head_size + kv_h * head_size;
            acc += scores[t] * vh[d];
        }
        outh[d] = acc;
    }
}

// ----------------------------------------------------------------------------
// cuBLAS FP16 Tensor Core matmul helper: out[d] = W[d x n] @ x[n]
//
// KEY CHANGE vs llama2_cublas_fp16.cu:
//   CUBLAS_COMPUTE_32F  →  CUBLAS_COMPUTE_16F
//
// With CUBLAS_COMPUTE_16F, cuBLAS accumulates partial sums in FP16 rather than
// FP32, selecting the fastest FP16 Tensor Core path.
//
// cuBLAS API constraint:
//   When computeType == CUBLAS_COMPUTE_16F, the output matrix C MUST be
//   CUDA_R_16F — CUDA_R_32F is not a valid combination and returns
//   CUBLAS_STATUS_NOT_SUPPORTED (error 15).  We therefore route the GEMM
//   output through a reusable FP16 scratch buffer and immediately widen back
//   to FP32 with a small conversion kernel so the rest of the pipeline stays
//   in FP32.  The two added kernel launches add negligible overhead vs the GEMM.
//
// alpha/beta scalars MUST be __half when computeType == CUBLAS_COMPUTE_16F.
//
// Precision impact:
//   For dim=4096 (LLaMA-2 7B), a single GEMV accumulates 4096 FP16 products.
//   Rounding error per step is ~2^-10 of the accumulated value vs ~2^-23 for
//   FP32 accumulation.  Over 4096 terms this produces an output error of
//   O(sqrt(4096) * 2^-10) ≈ 0.006, measurable with --verify.

__global__ void fp32_to_fp16_kernel(__half* __restrict__ out,
                                     const float* __restrict__ in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

/* Widen FP16 GEMM output back to FP32 activations */
__global__ void fp16_to_fp32_kernel(float* __restrict__ out,
                                     const __half* __restrict__ in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

/* Input activation scratch (FP32 → FP16 before GEMM) */
static __half*  d_x_fp16_scratch   = NULL;
static size_t   d_x_fp16_scratch_n = 0;

static void ensure_fp16_scratch(size_t n) {
    if (n > d_x_fp16_scratch_n) {
        if (d_x_fp16_scratch) cudaFree(d_x_fp16_scratch);
        CUDA_CHECK(cudaMalloc(&d_x_fp16_scratch, n * sizeof(__half)));
        d_x_fp16_scratch_n = n;
    }
}

/* Output scratch (FP16 GEMM result, widened back to FP32 afterwards) */
static __half*  d_y_fp16_scratch   = NULL;
static size_t   d_y_fp16_scratch_n = 0;

static void ensure_fp16_y_scratch(size_t n) {
    if (n > d_y_fp16_scratch_n) {
        if (d_y_fp16_scratch) cudaFree(d_y_fp16_scratch);
        CUDA_CHECK(cudaMalloc(&d_y_fp16_scratch, n * sizeof(__half)));
        d_y_fp16_scratch_n = n;
    }
}

static void cublas_sgemv_fp16tc(int n, int d,
                                 const __half* d_W,   /* d_W[n * d] row-major */
                                 const float*  d_x,   /* d_x[n] FP32 activations */
                                 float*        d_y)   /* d_y[d] FP32 output */
{
    /* Step 1: FP32 activation → FP16 input scratch */
    ensure_fp16_scratch(n);
    fp32_to_fp16_kernel<<<(n + 255) / 256, 256>>>(d_x_fp16_scratch, d_x, n);

    /* Step 2: GEMM — all operands FP16, compute FP16, output FP16.
     * CUBLAS_COMPUTE_16F requires Ctype == CUDA_R_16F; CUDA_R_32F output
     * is not supported and returns CUBLAS_STATUS_NOT_SUPPORTED (error 15). */
    ensure_fp16_y_scratch(d);
    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        d, 1, n,
        &alpha,
        d_W,              CUDA_R_16F, n,   /* A: weight matrix FP16 */
        d_x_fp16_scratch, CUDA_R_16F, n,   /* B: activation vector FP16 */
        &beta,
        d_y_fp16_scratch, CUDA_R_16F, d,   /* C: output FP16 (required by COMPUTE_16F) */
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT));

    /* Step 3: FP16 output → FP32 activation buffer so the rest of the
     * pipeline (RMSNorm, attention, SwiGLU) remains in FP32. */
    fp16_to_fp32_kernel<<<(d + 255) / 256, 256>>>(d_y, d_y_fp16_scratch, d);
}

// ----------------------------------------------------------------------------
// Memory management

static __half* alloc_half_gpu(size_t n) {
    __half* ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(__half)));
    return ptr;
}

/* Upload a FP32 CPU array as FP16 to GPU using a temporary buffer */
static __half* upload_fp32_as_fp16(const float* src, size_t n) {
    /* Convert on CPU */
    __half* tmp = (__half*)malloc(n * sizeof(__half));
    if (!tmp) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    for (size_t i = 0; i < n; i++) tmp[i] = __float2half(src[i]);
    __half* d_ptr = alloc_half_gpu(n);
    CUDA_CHECK(cudaMemcpy(d_ptr, tmp, n * sizeof(__half), cudaMemcpyHostToDevice));
    free(tmp);
    return d_ptr;
}

static void map_weights_from_ptr(CPUWeightsFP32* w, Config* p, float* ptr) {
    int head_size = p->dim / p->n_heads;
    unsigned long long nl = p->n_layers;

    w->token_embedding_table = ptr; ptr += (size_t)p->vocab_size * p->dim;
    w->rms_att_weight        = ptr; ptr += nl * p->dim;
    w->wq                    = ptr; ptr += nl * p->dim * (p->n_heads * head_size);
    w->wk                    = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wv                    = ptr; ptr += nl * p->dim * (p->n_kv_heads * head_size);
    w->wo                    = ptr; ptr += nl * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight        = ptr; ptr += nl * p->dim;
    w->w1                    = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->w2                    = ptr; ptr += nl * p->hidden_dim * p->dim;
    w->w3                    = ptr; ptr += nl * p->dim * p->hidden_dim;
    w->rms_final_weight      = ptr; ptr += p->dim;
    ptr += (size_t)p->seq_len * head_size / 2;
    ptr += (size_t)p->seq_len * head_size / 2;
    w->wcls = ptr;
}

static void upload_weights_to_gpu(GPUWeightsFP16* gw, CPUWeightsFP32* cw, Config* p) {
    int head_size = p->dim / p->n_heads;
    size_t nl = p->n_layers;
    printf("[fp16tc] Uploading weights to GPU as FP16 (compute: CUBLAS_COMPUTE_16F)...\n");

    gw->d_token_embedding_table = upload_fp32_as_fp16(cw->token_embedding_table,
                                                       (size_t)p->vocab_size * p->dim);
    gw->d_rms_att_weight = upload_fp32_as_fp16(cw->rms_att_weight, nl * p->dim);
    gw->d_wq = upload_fp32_as_fp16(cw->wq, nl * p->dim * (p->n_heads * head_size));
    gw->d_wk = upload_fp32_as_fp16(cw->wk, nl * p->dim * (p->n_kv_heads * head_size));
    gw->d_wv = upload_fp32_as_fp16(cw->wv, nl * p->dim * (p->n_kv_heads * head_size));
    gw->d_wo = upload_fp32_as_fp16(cw->wo, nl * (p->n_heads * head_size) * p->dim);
    gw->d_rms_ffn_weight = upload_fp32_as_fp16(cw->rms_ffn_weight, nl * p->dim);
    gw->d_w1 = upload_fp32_as_fp16(cw->w1, nl * p->dim * p->hidden_dim);
    gw->d_w2 = upload_fp32_as_fp16(cw->w2, nl * p->hidden_dim * p->dim);
    gw->d_w3 = upload_fp32_as_fp16(cw->w3, nl * p->dim * p->hidden_dim);
    gw->d_rms_final_weight = upload_fp32_as_fp16(cw->rms_final_weight, p->dim);

    if (cw->shared_weights) {
        gw->d_wcls = gw->d_token_embedding_table;
        gw->shared_weights = 1;
    } else {
        gw->d_wcls = upload_fp32_as_fp16(cw->wcls, (size_t)p->vocab_size * p->dim);
        gw->shared_weights = 0;
    }

    size_t fp32_mb = (size_t)p->vocab_size * p->dim * 4 / (1024*1024);
    size_t n_weights = (size_t)p->vocab_size * p->dim
        + nl * p->dim * 2
        + nl * p->dim * (p->n_heads * head_size)
        + nl * p->dim * (p->n_kv_heads * head_size) * 2
        + nl * p->dim * p->dim
        + nl * p->dim * p->hidden_dim * 3
        + p->dim;
    size_t fp16_mb = n_weights * 2 / (1024*1024);
    printf("[fp16tc] Weights: ~%zu MB FP32 -> ~%zu MB FP16 on GPU\n", fp32_mb, fp16_mb);
}

static void free_gpu_weights(GPUWeightsFP16* gw) {
    cudaFree(gw->d_token_embedding_table);
    cudaFree(gw->d_rms_att_weight);
    cudaFree(gw->d_rms_ffn_weight);
    cudaFree(gw->d_wq);
    cudaFree(gw->d_wk);
    cudaFree(gw->d_wv);
    cudaFree(gw->d_wo);
    cudaFree(gw->d_w1);
    cudaFree(gw->d_w2);
    cudaFree(gw->d_w3);
    cudaFree(gw->d_rms_final_weight);
    if (!gw->shared_weights) cudaFree(gw->d_wcls);
}

static void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    CUDA_CHECK(cudaMalloc(&s->d_x,           p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_xb,          p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_xb2,         p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_hb,          p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_hb2,         p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_q,           p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_k,           kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_v,           kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_key_cache,   (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_value_cache, (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_logits,      p->vocab_size * sizeof(float)));
    s->h_logits = (float*)malloc(p->vocab_size * sizeof(float));
    if (!s->h_logits) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
}

static void free_run_state(RunState* s) {
    cudaFree(s->d_x); cudaFree(s->d_xb); cudaFree(s->d_xb2);
    cudaFree(s->d_hb); cudaFree(s->d_hb2);
    cudaFree(s->d_q); cudaFree(s->d_k); cudaFree(s->d_v);
    cudaFree(s->d_key_cache); cudaFree(s->d_value_cache);
    cudaFree(s->d_logits);
    free(s->h_logits);
}

static void build_transformer(Transformer* t, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(EXIT_FAILURE); }
    if (fread(&t->config, sizeof(Config), 1, f) != 1) exit(EXIT_FAILURE);
    int shared = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    fseek(f, 0, SEEK_END);
    t->file_size = ftell(f);
    fclose(f);

#ifdef _WIN32
    t->fd = win_open(path, O_RDONLY);
    t->data = (float*)win_mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
#else
    t->fd = open(path, O_RDONLY);
    t->data = (float*)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
#endif
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    float* wptr = t->data + sizeof(Config)/sizeof(float);
    map_weights_from_ptr(&t->cpu_w, &t->config, wptr);
    t->cpu_w.shared_weights = shared;

    upload_weights_to_gpu(&t->gpu_w, &t->cpu_w, &t->config);
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer* t) {
    free_gpu_weights(&t->gpu_w);
    free_run_state(&t->state);
#ifdef _WIN32
    win_munmap(t->data, t->file_size);
    win_close(t->fd);
#else
    munmap(t->data, t->file_size);
    close(t->fd);
#endif
}

// ----------------------------------------------------------------------------
// Per-layer bottleneck profiler (FP16 Tensor Core variant)

static void profile_7b_bottlenecks(Transformer* t, int token, int pos) {
    Config*         p = &t->config;
    GPUWeightsFP16* w = &t->gpu_w;
    RunState*       s = &t->state;
    int dim        = p->dim;
    int kv_dim     = dim * p->n_kv_heads / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size  = dim / p->n_heads;
    float scale    = 1.0f / sqrtf((float)head_size);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float elapsed_ms, total_ms = 0.0f;
    float stage1_ms = 0.0f, stage2_ms = 0.0f, stage3_ms = 0.0f, stage4_ms = 0.0f;
    float stage5_ms = 0.0f, stage6_ms = 0.0f, stage7_ms = 0.0f, stage8_ms = 0.0f;

    printf("\n=== FP16TC MODEL BOTTLENECK ANALYSIS (CUBLAS_COMPUTE_16F, Layer 0, pos=%d) ===\n", pos);

    /* Prime d_x with the token embedding for layer 0 */
    {
        int threads = 256;
        int blocks  = (dim + threads - 1) / threads;
        embed_fp16_to_fp32<<<blocks, threads>>>(
            s->d_x, w->d_token_embedding_table, token * dim, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int l = 0; /* profile layer 0 only */

    /* --- Stage 1: Attention RMSNorm --- */
    cudaEventRecord(ev_start);
    cuda_rmsnorm_fp16(s->d_xb, s->d_x,
                      w->d_rms_att_weight + (size_t)l * dim, dim);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("1. RMSNorm (attention):        %.3f ms\n", elapsed_ms);
    stage1_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 2: QKV Projections (FP16 weights, FP16 accumulation -> FP32 output) --- */
    cudaEventRecord(ev_start);
    cublas_sgemv_fp16tc(dim, dim,
                        w->d_wq + (size_t)l * dim * dim,    s->d_xb, s->d_q);
    cublas_sgemv_fp16tc(dim, kv_dim,
                        w->d_wk + (size_t)l * dim * kv_dim, s->d_xb, s->d_k);
    cublas_sgemv_fp16tc(dim, kv_dim,
                        w->d_wv + (size_t)l * dim * kv_dim, s->d_xb, s->d_v);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("2. QKV Projections:            %.3f ms\n", elapsed_ms);
    stage2_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 3: RoPE + KV Cache --- */
    cudaEventRecord(ev_start);
    {
        int total   = p->n_heads * (head_size / 2);
        int threads = 256;
        int blocks  = (total + threads - 1) / threads;
        rope_kernel<<<blocks, threads>>>(
            s->d_q, s->d_k, p->n_heads, p->n_kv_heads, head_size, pos);
    }
    {
        size_t loff = (size_t)l * p->seq_len * kv_dim;
        CUDA_CHECK(cudaMemcpy(s->d_key_cache   + loff + (size_t)pos * kv_dim,
                              s->d_k, kv_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_value_cache + loff + (size_t)pos * kv_dim,
                              s->d_v, kv_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("3. RoPE + KV Cache:            %.3f ms\n", elapsed_ms);
    stage3_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 4: Flash Attention --- */
    cudaEventRecord(ev_start);
    {
        size_t loff = (size_t)l * p->seq_len * kv_dim;
        size_t smem = (size_t)(pos + 1) * sizeof(float);
        CUDA_CHECK(cudaMemset(s->d_xb, 0, dim * sizeof(float)));
        flash_attention_kernel<<<p->n_heads, 256, smem>>>(
            s->d_q,
            s->d_key_cache   + loff,
            s->d_value_cache + loff,
            s->d_xb,
            pos + 1, head_size,
            p->n_heads, p->n_kv_heads, scale);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("4. Flash Attention:            %.3f ms\n", elapsed_ms);
    stage4_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 5: Output Projection + Residual --- */
    cudaEventRecord(ev_start);
    cublas_sgemv_fp16tc(dim, dim,
                        w->d_wo + (size_t)l * dim * dim, s->d_xb, s->d_xb2);
    { int b = (dim + 255) / 256; add_kernel<<<b, 256>>>(s->d_x, s->d_xb2, dim); }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("5. Output Proj + Residual:     %.3f ms\n", elapsed_ms);
    stage5_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 6: FFN RMSNorm --- */
    cudaEventRecord(ev_start);
    cuda_rmsnorm_fp16(s->d_xb, s->d_x,
                      w->d_rms_ffn_weight + (size_t)l * dim, dim);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("6. RMSNorm (FFN):              %.3f ms\n", elapsed_ms);
    stage6_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 7: FFN W1 + W3 --- */
    cudaEventRecord(ev_start);
    cublas_sgemv_fp16tc(dim, hidden_dim,
                        w->d_w1 + (size_t)l * dim * hidden_dim, s->d_xb, s->d_hb);
    cublas_sgemv_fp16tc(dim, hidden_dim,
                        w->d_w3 + (size_t)l * dim * hidden_dim, s->d_xb, s->d_hb2);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("7. FFN W1+W3:                  %.3f ms\n", elapsed_ms);
    stage7_ms = elapsed_ms; total_ms += elapsed_ms;

    /* --- Stage 8: SwiGLU + W2 + Residual --- */
    cudaEventRecord(ev_start);
    { int b = (hidden_dim + 255) / 256;
      swiglu_kernel<<<b, 256>>>(s->d_hb, s->d_hb2, hidden_dim); }
    cublas_sgemv_fp16tc(hidden_dim, dim,
                        w->d_w2 + (size_t)l * hidden_dim * dim, s->d_hb, s->d_xb);
    { int b = (dim + 255) / 256; add_kernel<<<b, 256>>>(s->d_x, s->d_xb, dim); }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    printf("8. SwiGLU + W2:                %.3f ms\n", elapsed_ms);
    stage8_ms = elapsed_ms; total_ms += elapsed_ms;

    printf("\nTOTAL LAYER TIME:        %.3f ms\n", total_ms);
    float est_32_layer_ms = total_ms * p->n_layers;
    float est_tok_s       = est_32_layer_ms > 0.0f ? (1000.0f / est_32_layer_ms) : 0.0f;
    printf("ESTIMATED %d-LAYER TIME: %.1f ms = %.1f tok/s\n",
           p->n_layers, est_32_layer_ms, est_tok_s);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    g_profile_triggered = true;
    append_profile_metrics_csv(
        token, pos, p->dim, p->hidden_dim, p->n_heads, p->n_kv_heads,
        stage1_ms, stage2_ms, stage3_ms, stage4_ms,
        stage5_ms, stage6_ms, stage7_ms, stage8_ms,
        total_ms, est_32_layer_ms, est_tok_s);
    printf("Profiling metrics appended to %s\n", g_profile_csv_path);
}

// ----------------------------------------------------------------------------
// GPU forward pass (FP16 weights, FP16 accumulation, FP32 activations)

static float* forward_gpu(Transformer* t, int token, int pos) {
    Config* p  = &t->config;
    GPUWeightsFP16* w = &t->gpu_w;
    RunState* s = &t->state;

    int dim        = p->dim;
    int kv_dim     = dim * p->n_kv_heads / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size  = dim / p->n_heads;
    float scale    = 1.0f / sqrtf((float)head_size);

    /* Profiling: run once at the requested token position */
    if (g_enable_profile && pos == g_profile_pos && !g_profile_triggered)
        profile_7b_bottlenecks(t, token, pos);

    /* 1. Token embedding: copy FP16 row to FP32 d_x */
    {
        int threads = 256;
        int blocks  = (dim + threads - 1) / threads;
        embed_fp16_to_fp32<<<blocks, threads>>>(
            s->d_x, w->d_token_embedding_table, token * dim, dim);
    }

    for (int l = 0; l < p->n_layers; l++) {
        /* 2. Attention RMSNorm */
        cuda_rmsnorm_fp16(s->d_xb, s->d_x, w->d_rms_att_weight + (size_t)l * dim, dim);

        /* 3. QKV projections (FP16 compute) */
        cublas_sgemv_fp16tc(dim, dim,    w->d_wq + (size_t)l * dim * dim,               s->d_xb, s->d_q);
        cublas_sgemv_fp16tc(dim, kv_dim, w->d_wk + (size_t)l * dim * kv_dim,            s->d_xb, s->d_k);
        cublas_sgemv_fp16tc(dim, kv_dim, w->d_wv + (size_t)l * dim * kv_dim,            s->d_xb, s->d_v);

        /* 4. RoPE */
        {
            int total = p->n_heads * (head_size / 2);
            int threads = 256;
            int blocks  = (total + threads - 1) / threads;
            rope_kernel<<<blocks, threads>>>(s->d_q, s->d_k,
                p->n_heads, p->n_kv_heads, head_size, pos);
        }

        /* 5. KV cache store */
        {
            size_t loff = (size_t)l * p->seq_len * kv_dim;
            CUDA_CHECK(cudaMemcpy(s->d_key_cache   + loff + (size_t)pos * kv_dim,
                                  s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(s->d_value_cache + loff + (size_t)pos * kv_dim,
                                  s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        /* 6. Flash Attention */
        {
            size_t loff = (size_t)l * p->seq_len * kv_dim;
            size_t smem = (size_t)(pos + 1) * sizeof(float);
            CUDA_CHECK(cudaMemset(s->d_xb, 0, dim * sizeof(float)));
            flash_attention_kernel<<<p->n_heads, 256, smem>>>(
                s->d_q,
                s->d_key_cache   + loff,
                s->d_value_cache + loff,
                s->d_xb,
                pos + 1, head_size,
                p->n_heads, p->n_kv_heads, scale);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        /* 7. Output projection + residual */
        cublas_sgemv_fp16tc(dim, dim, w->d_wo + (size_t)l * dim * dim, s->d_xb, s->d_xb2);
        { int b = (dim+255)/256; add_kernel<<<b,256>>>(s->d_x, s->d_xb2, dim); }

        /* 8. FFN RMSNorm */
        cuda_rmsnorm_fp16(s->d_xb, s->d_x, w->d_rms_ffn_weight + (size_t)l * dim, dim);

        /* 9. FFN W1, W3 */
        cublas_sgemv_fp16tc(dim, hidden_dim, w->d_w1 + (size_t)l * dim * hidden_dim, s->d_xb, s->d_hb);
        cublas_sgemv_fp16tc(dim, hidden_dim, w->d_w3 + (size_t)l * dim * hidden_dim, s->d_xb, s->d_hb2);

        /* 10. SwiGLU */
        { int b = (hidden_dim+255)/256; swiglu_kernel<<<b,256>>>(s->d_hb, s->d_hb2, hidden_dim); }

        /* 11. FFN W2 + residual */
        cublas_sgemv_fp16tc(hidden_dim, dim, w->d_w2 + (size_t)l * hidden_dim * dim, s->d_hb, s->d_xb);
        { int b = (dim+255)/256; add_kernel<<<b,256>>>(s->d_x, s->d_xb, dim); }
    }

    /* 12. Final RMSNorm */
    cuda_rmsnorm_fp16(s->d_x, s->d_x, w->d_rms_final_weight, dim);

    /* 13. Classifier */
    cublas_sgemv_fp16tc(dim, p->vocab_size, w->d_wcls, s->d_x, s->d_logits);

    CUDA_CHECK(cudaMemcpy(s->h_logits, s->d_logits,
                          p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s->h_logits;
}

// ----------------------------------------------------------------------------
// CPU FP16 forward pass (--verify mode only; CPU still accumulates in FP32)

static float cpu_fp16_to_float(__half v) { return __half2float(v); }

static void build_cpu_fp16_weights(CPUWeightsFP16* cw, CPUWeightsFP32* src, Config* p) {
    int head_size = p->dim / p->n_heads;
    size_t nl = p->n_layers;

    auto cvt = [](const float* from, size_t n) -> __half* {
        __half* buf = (__half*)malloc(n * sizeof(__half));
        if (!buf) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
        for (size_t i = 0; i < n; i++) buf[i] = __float2half(from[i]);
        return buf;
    };

    cw->token_embedding_table = cvt(src->token_embedding_table, (size_t)p->vocab_size * p->dim);
    cw->rms_att_weight  = cvt(src->rms_att_weight,  nl * p->dim);
    cw->rms_ffn_weight  = cvt(src->rms_ffn_weight,  nl * p->dim);
    cw->wq = cvt(src->wq, nl * p->dim * (p->n_heads * head_size));
    cw->wk = cvt(src->wk, nl * p->dim * (p->n_kv_heads * head_size));
    cw->wv = cvt(src->wv, nl * p->dim * (p->n_kv_heads * head_size));
    cw->wo = cvt(src->wo, nl * (p->n_heads * head_size) * p->dim);
    cw->w1 = cvt(src->w1, nl * p->dim * p->hidden_dim);
    cw->w2 = cvt(src->w2, nl * p->hidden_dim * p->dim);
    cw->w3 = cvt(src->w3, nl * p->dim * p->hidden_dim);
    cw->rms_final_weight = cvt(src->rms_final_weight, p->dim);
    cw->shared_weights = src->shared_weights;
    if (src->shared_weights)
        cw->wcls = cw->token_embedding_table;
    else
        cw->wcls = cvt(src->wcls, (size_t)p->vocab_size * p->dim);
}

static void malloc_cpu_run_state(CPURunState* s, Config* p) {
    int kv_dim = p->dim * p->n_kv_heads / p->n_heads;
    s->x           = (float*)calloc(p->dim, sizeof(float));
    s->xb          = (float*)calloc(p->dim, sizeof(float));
    s->xb2         = (float*)calloc(p->dim, sizeof(float));
    s->hb          = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2         = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q           = (float*)calloc(p->dim, sizeof(float));
    s->k           = (float*)calloc(kv_dim, sizeof(float));
    s->v           = (float*)calloc(kv_dim, sizeof(float));
    s->att         = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits      = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache   = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

static void cpu_rmsnorm_fp16(float* o, const float* x, const __half* w, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = ss / size + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++)
        o[j] = cpu_fp16_to_float(w[j]) * (ss * x[j]);
}

static void cpu_matmul_fp16(float* xout, const float* x, const __half* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) val += cpu_fp16_to_float(w[i*n+j]) * x[j];
        xout[i] = val;
    }
}

static float* forward_cpu_fp16(CPUWeightsFP16* w, CPURunState* s, Config* p,
                                int token, int pos) {
    int dim        = p->dim;
    int kv_dim     = dim * p->n_kv_heads / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size  = dim / p->n_heads;
    float* x = s->x;

    /* embedding */
    const __half* row = w->token_embedding_table + (size_t)token * dim;
    for (int i = 0; i < dim; i++) x[i] = cpu_fp16_to_float(row[i]);

    for (int l = 0; l < p->n_layers; l++) {
        cpu_rmsnorm_fp16(s->xb, x, w->rms_att_weight + (size_t)l*dim, dim);

        cpu_matmul_fp16(s->q, s->xb, w->wq + (size_t)l*dim*dim, dim, dim);
        cpu_matmul_fp16(s->k, s->xb, w->wk + (size_t)l*dim*kv_dim, dim, kv_dim);
        cpu_matmul_fp16(s->v, s->xb, w->wv + (size_t)l*dim*kv_dim, dim, kv_dim);

        /* RoPE */
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
                float val  = pos * freq;
                float fcr = cosf(val), fci = sinf(val);
                float q0 = q[i], q1 = q[i+1];
                q[i]   = q0*fcr - q1*fci;
                q[i+1] = q0*fci + q1*fcr;
            }
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * head_size;
            for (int i = 0; i < head_size; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
                float val  = pos * freq;
                float fcr = cosf(val), fci = sinf(val);
                float k0 = k[i], k1 = k[i+1];
                k[i]   = k0*fcr - k1*fci;
                k[i+1] = k0*fci + k1*fcr;
            }
        }

        /* KV cache */
        size_t loff = (size_t)l * p->seq_len * kv_dim;
        memcpy(s->key_cache   + loff + (size_t)pos*kv_dim, s->k, kv_dim*sizeof(float));
        memcpy(s->value_cache + loff + (size_t)pos*kv_dim, s->v, kv_dim*sizeof(float));

        /* Multi-head attention */
        for (int h = 0; h < p->n_heads; h++) {
            float* q  = s->q  + h * head_size;
            float* att = s->att + h * p->seq_len;
            int kv_h   = h * p->n_kv_heads / p->n_heads;
            float sc   = 1.0f / sqrtf((float)head_size);
            float mx   = -1e30f;
            for (int t = 0; t <= pos; t++) {
                const float* k = s->key_cache + loff + (size_t)t*kv_dim + kv_h*head_size;
                float sc2 = 0.0f;
                for (int d = 0; d < head_size; d++) sc2 += q[d] * k[d];
                att[t] = sc2 * sc;
                if (att[t] > mx) mx = att[t];
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) { att[t] = expf(att[t]-mx); sum += att[t]; }
            for (int t = 0; t <= pos; t++) att[t] /= sum;
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* v = s->value_cache + loff + (size_t)t*kv_dim + kv_h*head_size;
                for (int d = 0; d < head_size; d++) xb[d] += att[t] * v[d];
            }
        }

        cpu_matmul_fp16(s->xb2, s->xb, w->wo + (size_t)l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        cpu_rmsnorm_fp16(s->xb, x, w->rms_ffn_weight + (size_t)l*dim, dim);
        cpu_matmul_fp16(s->hb,  s->xb, w->w1 + (size_t)l*dim*hidden_dim, dim, hidden_dim);
        cpu_matmul_fp16(s->hb2, s->xb, w->w3 + (size_t)l*dim*hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float v = s->hb[i];
            v = v / (1.0f + expf(-v));
            s->hb[i] = v * s->hb2[i];
        }
        cpu_matmul_fp16(s->xb, s->hb, w->w2 + (size_t)l*hidden_dim*dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    cpu_rmsnorm_fp16(x, x, w->rms_final_weight, dim);
    cpu_matmul_fp16(s->logits, x, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct { char* str; int id; } TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((const TokenIndex*)a)->str, ((const TokenIndex*)b)->str);
}

static void build_tokenizer(Tokenizer* t, const char* path, int vocab_size) {
    t->vocab_size   = vocab_size;
    t->vocab        = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i*2]   = (unsigned char)i;
        t->byte_pieces[i*2+1] = '\0';
    }
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open tokenizer %s\n", path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) exit(EXIT_FAILURE);
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, f) != 1) exit(EXIT_FAILURE);
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) exit(EXIT_FAILURE);
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, f) != 1) exit(EXIT_FAILURE);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

static void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

static char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
        piece = (char*)t->byte_pieces + byte_val * 2;
    return piece;
}

static void safe_printf(const char* piece) {
    if (!piece || !piece[0]) return;
    if (piece[1] == '\0') {
        unsigned char b = piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

static int str_lookup(const char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok; tok.str = (char*)str;
    TokenIndex* res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size,
                                            sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

static void encode(Tokenizer* t, const char* text, int8_t bos, int8_t eos,
                   int* tokens, int* n_tokens) {
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++)
            { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* buf = (char*)malloc(t->max_token_length * 2 + 3);
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0]) {
        int dummy = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy != -1) tokens[(*n_tokens)++] = dummy;
    }
    for (const char* c = text; *c; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        buf[str_len++] = *c; buf[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
        if (id != -1) { tokens[(*n_tokens)++] = id; }
        else { for (size_t i = 0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)buf[i] + 3; }
        str_len = 0;
    }
    while (1) {
        float best = -1e10f; int best_id = -1, best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            snprintf(buf, t->max_token_length*2+3, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(buf, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best)
                { best = t->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < *n_tokens-1; i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(buf);
}

// ----------------------------------------------------------------------------
// Sampler

typedef struct { float prob; int index; } ProbIndex;
typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static unsigned int random_u32(unsigned long long* s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (*s * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long* s) { return (random_u32(s) >> 8) / 16777216.0f; }

static int compare_probs(const void* a, const void* b) {
    const ProbIndex* pa = (const ProbIndex*)a;
    const ProbIndex* pb = (const ProbIndex*)b;
    return (pa->prob > pb->prob) ? -1 : (pa->prob < pb->prob) ? 1 : 0;
}

static int sample_argmax(const float* p, int n) {
    int mi = 0; for (int i = 1; i < n; i++) if (p[i] > p[mi]) mi = i; return mi;
}

static int sample_topp(float* p, int n, float topp, ProbIndex* pi, float coin) {
    float cutoff = (1.0f - topp) / (n - 1);
    int n0 = 0;
    for (int i = 0; i < n; i++) if (p[i] >= cutoff) { pi[n0].index=i; pi[n0].prob=p[i]; n0++; }
    qsort(pi, n0, sizeof(ProbIndex), compare_probs);
    float cum = 0.0f; int last = n0-1;
    for (int i = 0; i < n0; i++) { cum += pi[i].prob; if (cum > topp) { last=i; break; } }
    float r = coin * cum, cdf = 0.0f;
    for (int i = 0; i <= last; i++) { cdf += pi[i].prob; if (r < cdf) return pi[i].index; }
    return pi[last].index;
}

static void softmax(float* x, int n) {
    float mx = x[0]; for (int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float sum = 0.0f;
    for (int i=0;i<n;i++) { x[i]=expf(x[i]-mx); sum+=x[i]; }
    for (int i=0;i<n;i++) x[i]/=sum;
}

static int do_sample(Sampler* s, float* logits) {
    if (s->temperature < 1e-6f) return sample_argmax(logits, s->vocab_size);
    for (int i=0;i<s->vocab_size;i++) logits[i] /= s->temperature;
    softmax(logits, s->vocab_size);
    float coin = random_f32(&s->rng_state);
    if (s->topp <= 0.0f || s->topp >= 1.0f) {
        float cdf = 0.0f;
        for (int i=0;i<s->vocab_size;i++) { cdf += logits[i]; if (coin < cdf) return i; }
        return s->vocab_size - 1;
    }
    return sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
}

static void build_sampler(Sampler* s, int vocab_size, float temp, float topp, unsigned long long seed) {
    s->vocab_size = vocab_size; s->temperature = temp; s->topp = topp;
    s->rng_state  = seed;
    s->probindex  = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
}
static void free_sampler(Sampler* s) { free(s->probindex); }

// ----------------------------------------------------------------------------
// Timing

static long long time_in_ms(void) {
#ifdef _WIN32
    return win_time_in_ms();
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

// ----------------------------------------------------------------------------
// Generation loop

static void generate(Transformer* t, Tokenizer* tok, Sampler* sampler,
                     const char* prompt, int steps,
                     bool do_verify, CPUWeightsFP16* cpu_w, CPURunState* cpu_s)
{
    Config* p = &t->config;

    int num_prompt_tokens = 0;
    size_t prompt_len = prompt ? strlen(prompt) : 0;
    int* prompt_tokens = (int*)malloc((prompt_len + 3) * sizeof(int));
    if (prompt) encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { prompt_tokens[0] = 1; num_prompt_tokens = 1; }

    float* gpu_logits_copy = do_verify ? (float*)malloc(p->vocab_size * sizeof(float)) : NULL;

    long long start = 0;
    int token = prompt_tokens[0], next, pos = 0;
#ifdef DUMP_LOGITS
    int logit_step = 0;       /* increments each time a logit vector is written */
    int logit_dump_count = 0; /* increments each time a token ID is written (for JSON format) */
#endif

    while (pos < steps) {
        float* gpu_logits = forward_gpu(t, token, pos);

        if (do_verify) {
            memcpy(gpu_logits_copy, gpu_logits, p->vocab_size * sizeof(float));
            float* cpu_logits = forward_cpu_fp16(cpu_w, cpu_s, p, token, pos);
            float max_diff = 0.0f, mean_diff = 0.0f;
            for (int i = 0; i < p->vocab_size; i++) {
                float d = fabsf(cpu_logits[i] - gpu_logits_copy[i]);
                if (d > max_diff) max_diff = d;
                mean_diff += d;
            }
            mean_diff /= p->vocab_size;
            /* Note: max_diff will be larger here than with COMPUTE_32F because
             * the GPU accumulates in FP16 while the CPU reference uses FP32. */
            fprintf(stderr, "[verify pos=%3d] max_diff=%.5f  mean_diff=%.6f%s\n",
                    pos, max_diff, mean_diff,
                    (max_diff > 1.0f) ? "  *** LARGE DIVERGENCE ***" : "");
        }
#ifdef DUMP_LOGITS
        if (g_logit_bin_path[0] != '\0' && pos >= num_prompt_tokens - 1) {
            FILE* _lf = fopen(g_logit_bin_path, logit_step == 0 ? "wb" : "ab");
            if (_lf) {
                fwrite(gpu_logits, sizeof(float), p->vocab_size, _lf);
                fclose(_lf);
            }
            logit_step++;
        }
#endif

        if (pos < num_prompt_tokens - 1)
            next = prompt_tokens[pos + 1];
        else
            next = do_sample(sampler, gpu_logits);
        pos++;
#ifdef DUMP_LOGITS
        if (g_token_ids_path[0] != '\0' && (pos - 1) >= num_prompt_tokens - 1) {
            FILE* _tf = fopen(g_token_ids_path, logit_dump_count == 0 ? "w" : "a");
            if (_tf) { fprintf(_tf, logit_dump_count == 0 ? "[%d" : ",%d", next); fclose(_tf); }
            logit_dump_count++;
        }
#endif

        if (next == 2) { printf("\n"); break; }

        char* piece = decode(tok, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) start = time_in_ms();
    }
    printf("\n");

    if (pos > 1) {
        long long end = time_in_ms();
        double tok_s = (pos - 1) / ((end - start) / 1000.0);
        fprintf(stderr, "Achieved tok/s: %.2f\n", tok_s);
    }

    free(prompt_tokens);
    if (gpu_logits_copy) free(gpu_logits_copy);
#ifdef DUMP_LOGITS
    if (g_token_ids_path[0] != '\0' && logit_dump_count > 0) {
        FILE* _tf = fopen(g_token_ids_path, "a");
        if (_tf) { fprintf(_tf, "]"); fclose(_tf); }
    }
#endif
}

// ----------------------------------------------------------------------------
// Main

static void error_usage(void) {
    fprintf(stderr, "Usage: llama2_cublas_fp16tc <checkpoint.bin> [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>   temperature (default 1.0)\n");
    fprintf(stderr, "  -p <float>   top-p (default 0.9)\n");
    fprintf(stderr, "  -s <int>     random seed (default: time)\n");
    fprintf(stderr, "  -n <int>     number of steps (default 256)\n");
    fprintf(stderr, "  -i <string>  prompt\n");
    fprintf(stderr, "  -z <string>  tokenizer path (default: tokenizer.bin)\n");
    fprintf(stderr, "  -P <int>     profile layer-0 bottlenecks at this token pos (default: off)\n");
    fprintf(stderr, "  -R <string>  profiling CSV output path (default: cublas_fp16tc_profile_metrics.csv)\n");
    fprintf(stderr, "  --verify     compare GPU logits with CPU FP16 each token\n");
    fprintf(stderr, "\nNote: CUBLAS_COMPUTE_16F means FP16 accumulation — expect larger\n");
    fprintf(stderr, "      logit diffs vs --verify than the cublas_fp16 (COMPUTE_32F) baseline.\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 2) error_usage();

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path  = "tokenizer.bin";
    float temperature  = 1.0f;
    float topp         = 0.9f;
    int   steps        = 256;
    char* prompt       = NULL;
    unsigned long long rng_seed = (unsigned long long)time(NULL);
    bool  do_verify    = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) {
            do_verify = true;
        } else if (i + 1 < argc) {
            if      (strcmp(argv[i], "-t") == 0) temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "-p") == 0) topp        = atof(argv[++i]);
            else if (strcmp(argv[i], "-s") == 0) rng_seed    = (unsigned long long)atoi(argv[++i]);
            else if (strcmp(argv[i], "-n") == 0) steps       = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0) prompt      = argv[++i];
            else if (strcmp(argv[i], "-z") == 0) tokenizer_path = argv[++i];
            else if (strcmp(argv[i], "-P") == 0) {
                g_enable_profile = true;
                g_profile_pos    = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-R") == 0) {
                g_enable_profile = true;
                strncpy(g_profile_csv_path, argv[++i], sizeof(g_profile_csv_path) - 1);
                g_profile_csv_path[sizeof(g_profile_csv_path) - 1] = '\0';
            }
#ifdef DUMP_LOGITS
            else if (strcmp(argv[i], "-L") == 0) {
                strncpy(g_logit_bin_path, argv[++i], sizeof(g_logit_bin_path) - 1);
                g_logit_bin_path[sizeof(g_logit_bin_path) - 1] = '\0';
            }
            else if (strcmp(argv[i], "-T") == 0) {
                strncpy(g_token_ids_path, argv[++i], sizeof(g_token_ids_path) - 1);
                g_token_ids_path[sizeof(g_token_ids_path) - 1] = '\0';
            }
#endif
            else error_usage();
        } else { error_usage(); }
    }
    if (steps == 0) steps = 256;

    /* Initialize CUDA */
    int deviceCount = 0;
    cudaError_t cst = cudaGetDeviceCount(&deviceCount);
    if (cst != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found: %s\n", cudaGetErrorString(cst));
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        fprintf(stderr, "[fp16tc] GPU: %s (CC %d.%d, %.1f GB)\n", prop.name,
                prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        fprintf(stderr, "[fp16tc] Compute mode: CUBLAS_COMPUTE_16F (FP16 accumulation)\n");
    }
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    /* Load model */
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    /* Load tokenizer */
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    /* Build sampler */
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    /* Optionally build CPU FP16 state for verification */
    CPUWeightsFP16 cpu_w;
    CPURunState    cpu_s;
    if (do_verify) {
        fprintf(stderr, "[verify] Building CPU FP16 weight copy (CPU accumulates FP32)...\n");
        build_cpu_fp16_weights(&cpu_w, &transformer.cpu_w, &transformer.config);
        malloc_cpu_run_state(&cpu_s, &transformer.config);
        fprintf(stderr, "[verify] Ready. Diffs vs GPU will reflect COMPUTE_16F vs COMPUTE_32F gap.\n");
    }

    /* Run */
    generate(&transformer, &tokenizer, &sampler,
             prompt, steps,
             do_verify,
             do_verify ? &cpu_w : NULL,
             do_verify ? &cpu_s : NULL);

    /* Cleanup */
    if (do_verify) {
        free(cpu_w.token_embedding_table);
        free(cpu_w.rms_att_weight); free(cpu_w.rms_ffn_weight);
        free(cpu_w.wq); free(cpu_w.wk); free(cpu_w.wv); free(cpu_w.wo);
        free(cpu_w.w1); free(cpu_w.w2); free(cpu_w.w3);
        free(cpu_w.rms_final_weight);
        if (!cpu_w.shared_weights) free(cpu_w.wcls);
        free(cpu_s.x); free(cpu_s.xb); free(cpu_s.xb2);
        free(cpu_s.hb); free(cpu_s.hb2);
        free(cpu_s.q); free(cpu_s.k); free(cpu_s.v);
        free(cpu_s.att); free(cpu_s.logits);
        free(cpu_s.key_cache); free(cpu_s.value_cache);
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    if (d_x_fp16_scratch) cudaFree(d_x_fp16_scratch);
    if (d_y_fp16_scratch) cudaFree(d_y_fp16_scratch);
    cublasDestroy(cublas_handle);
    return 0;
}
