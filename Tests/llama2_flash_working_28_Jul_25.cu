/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>

#if defined _WIN32
    #include <io.h> 
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define USE_CUDA 1
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define VECTORIZE_WIDTH 4
#define TILE_SIZE 32





// // Global variables for CUDA
// static cublasHandle_t cublas_handle;
static bool g_cuda_available = false;
static FILE* g_log_file = NULL;

void init_logging(const char* filename) {
    g_log_file = fopen(filename, "w");
    if (!g_log_file) {
        fprintf(stderr, "Warning: Could not open log file %s\n", filename);
    }
}

void set_cuda_availability(bool available) {
    g_cuda_available = available;
}

void close_logging() {
    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = NULL;
    }
}

// void set_cuda_availability(bool available) {
//     g_cuda_available = available;
//     if (available) {
//         cublasCreate(&cublas_handle);
//     }
// }


// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;

    // CUDA device pointers
    float* d_token_embedding_table;
    float* d_rms_att_weight;
    float* d_rms_ffn_weight;
    float* d_wq, *d_wk, *d_wv, *d_wo;
    float* d_w1, *d_w2, *d_w3;
    float* d_rms_final_weight;
    float* d_wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k_original; // key original buffer
    float *v_original; // value original buffer
    float *k; // key (pointer, may point to cache)
    float *v; // value (pointer, may point to cache)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)

    // CUDA device pointers
    float *d_x, *d_xb, *d_xb2, *d_hb, *d_hb2;
    float *d_q, *d_k, *d_v, *d_att, *d_logits;
    float *d_key_cache, *d_value_cache;
    float *d_output, *d_temp_storage;

} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

// ----------------------------------------------------------------------------
// CUDA Kernels

// Add custom CUDA kernels to replace cuBLAS operations

// Matrix-vector multiplication kernel (replaces cublasSgemv)
__global__ void sgemv_kernel(float* y, const float* A, const float* x, int m, int n, float alpha, float beta, bool transpose) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for vector reduction
    extern __shared__ float smem[];
    
    if (transpose) {
        // A^T * x: A is m x n stored in column-major, A^T is n x m
        // We want y[i] = sum_j(A^T[i,j] * x[j]) = sum_j(A[j,i] * x[j])
        if (idx < n) {
            float sum = 0.0f;
            
            // Process in chunks to improve memory coalescing
            for (int j = 0; j < m; j += 32) {
                int end_j = min(j + 32, m);
                for (int jj = j; jj < end_j; jj++) {
                    sum += A[idx * m + jj] * x[jj];  // A[jj,idx] in column-major = A[idx*m + jj]
                }
            }
            y[idx] = alpha * sum + beta * y[idx];
        }
    } else {
        // A * x: y[i] = sum(A[i,j] * x[j]) for j in [0,n)
        // Use block-level reduction for better performance
        if (blockIdx.x < m) {
            int row = blockIdx.x;
            float sum = 0.0f;
            
            // Each thread processes multiple elements
            for (int j = tid; j < n; j += blockDim.x) {
                sum += A[j * m + row] * x[j];  // A[row,j] in column-major = A[j*m + row]
            }
            
            // Store partial sum in shared memory
            smem[tid] = sum;
            __syncthreads();
            
            // Reduce within block
            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    smem[tid] += smem[tid + stride];
                }
                __syncthreads();
            }
            
            // Write result
            if (tid == 0) {
                y[row] = alpha * smem[0] + beta * y[row];
            }
        }
    }
}

// Vector addition kernel (replaces cublasSaxpy)
__global__ void saxpy_kernel(float* y, const float* x, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

// Wrapper functions to replace cuBLAS calls
void cuda_sgemv(float* y, const float* A, const float* x, int m, int n, float alpha, float beta, bool transpose) {
    dim3 blockDim(256);
    dim3 gridDim;
    size_t smem_size = 0;
    
    if (transpose) {
        // For transpose=true: A^T * x, output size is n
        // Use thread-per-output parallelization
        int output_size = n;
        gridDim.x = (output_size + blockDim.x - 1) / blockDim.x;
    } else {
        // For transpose=false: A * x, output size is m
        // Use block-per-row parallelization for better performance
        gridDim.x = m;  // One block per output row
        smem_size = blockDim.x * sizeof(float);  // Shared memory for reduction
    }
    
    sgemv_kernel<<<gridDim, blockDim, smem_size>>>(y, A, x, m, n, alpha, beta, transpose);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_saxpy(float* y, const float* x, float alpha, int n) {
    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    saxpy_kernel<<<gridDim, blockDim>>>(y, x, alpha, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// RMSNorm CUDA kernel
__global__ void rmsnorm_kernel(float* output, float* input, float* weight, int size, float eps, int layer, int pos, int print_debug) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for reduction
    __shared__ double shared_sum[32];
    
    // Compute sum of squares
    double sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        double val = (double) input[i];
        sum += val * val;
    }
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // Store warp results in shared memory
    if (threadIdx.x % 32 == 0) {
        shared_sum[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    
    // Final reduction
    if (threadIdx.x < 32) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared_sum[threadIdx.x] : 0.0f;
        sum = warpReduceSum(sum);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        shared_sum[0] = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    
    float rms_scale = shared_sum[0];

    // if (pos < 10 && layer == 0 && print_debug && idx == 0) {
    //     printf("print_debug rms_scale: %f\n", rms_scale);
    // }

    // __syncthreads();
    
    // Apply normalization
    if (idx < size) {
        output[idx] = input[idx] * rms_scale * weight[idx];
    }
}

// SwiGLU activation CUDA kernel
__global__ void swiglu_kernel(float* output, float* gate, float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = gate[idx];
        float silu = x / (1.0f + expf(-x)); // SiLU activation
        output[idx] = silu * up[idx];
    }
}

// Softmax CUDA kernel
__global__ void softmax_kernel(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    // Find maximum
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }
    max_val = warpReduceMax(max_val);
    
    if (threadIdx.x % 32 == 0) {
        shared_max[threadIdx.x / 32] = max_val;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        max_val = (threadIdx.x < blockDim.x / 32) ? shared_max[threadIdx.x] : -INFINITY;
        max_val = warpReduceMax(max_val);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        shared_max[0] = max_val;
    }
    __syncthreads();
    
    max_val = shared_max[0];
    
    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }
    sum = warpReduceSum(sum);
    
    if (threadIdx.x % 32 == 0) {
        shared_sum[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        sum = (threadIdx.x < blockDim.x / 32) ? shared_sum[threadIdx.x] : 0.0f;
        sum = warpReduceSum(sum);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        shared_sum[0] = sum;
    }
    __syncthreads();
    
    sum = shared_sum[0];
    
    // Normalize
    if (idx < size) {
        output[idx] = output[idx] / sum;
    }
}

// Flash Attention CUDA kernel (simplified version)
// Fix 2: Corrected CUDA Flash Attention kernel with proper indexing
__global__ void flash_attention_kernel_fixed(
    float* output,           // [n_heads, head_dim]
    float* query,            // [n_heads, head_dim]  
    float* key_cache,        // [seq_len, n_kv_heads, head_dim]
    float* value_cache,      // [seq_len, n_kv_heads, head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float scale
) {
    int head_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (head_idx >= n_heads || dim_idx >= head_dim) return;
    
    // Calculate which KV head to use
    int kv_head_idx = head_idx * n_kv_heads / n_heads;
    
    // Shared memory for scores
    extern __shared__ float shared_scores[];
    
    float output_val = 0.0f;
    float max_score = -INFINITY;
    
    // Phase 1: Find maximum score
    for (int t = 0; t <= pos; t++) {
        if (dim_idx == 0) {  // Only one thread computes the score
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = query[head_idx * head_dim + d];
                float k_val = key_cache[t * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
                score += q_val * k_val;
            }
            score *= scale;
            shared_scores[t] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    
    // Broadcast max_score to all threads
    if (dim_idx == 0) {
        shared_scores[pos + 1] = max_score;  // Store max_score at the end
    }
    __syncthreads();
    max_score = shared_scores[pos + 1];
    
    // Phase 2: Compute softmax
    float sum_exp = 0.0f;
    if (dim_idx == 0) {
        for (int t = 0; t <= pos; t++) {
            shared_scores[t] = expf(shared_scores[t] - max_score);
            sum_exp += shared_scores[t];
        }
        shared_scores[pos + 2] = sum_exp;  // Store sum_exp
    }
    __syncthreads();
    sum_exp = shared_scores[pos + 2];
    
    // Phase 3: Apply attention weights
    for (int t = 0; t <= pos; t++) {
        float weight = shared_scores[t] / sum_exp;
        float v_val = value_cache[t * n_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx];
        output_val += weight * v_val;
    }
    
    // Write output
    output[head_idx * head_dim + dim_idx] = output_val;
}


// // RoPE (Rotary Position Embedding) kernel
// __global__ void rope_kernel(
//     float* q, float* k,
//     int batch_size, int seq_len, int n_heads, int head_dim,
//     int pos
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_elements = batch_size * seq_len * n_heads * head_dim;
    
//     if (idx >= total_elements) return;
    
//     int dim_idx = idx % head_dim;
//     int head_idx = (idx / head_dim) % n_heads;
//     int seq_idx = (idx / (head_dim * n_heads)) % seq_len;
//     int batch_idx = idx / (head_dim * n_heads * seq_len);
    
//     if (seq_idx != pos) return; // Only apply to current position
    
//     // Apply RoPE only to even dimensions (pairs)
//     if (dim_idx % 2 == 0 && dim_idx + 1 < head_dim) {
//         float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
//         float angle = pos * freq;
//         float cos_val = cosf(angle);
//         float sin_val = sinf(angle);
        
//         int even_idx = idx;
//         int odd_idx = idx + 1;
        
//         float q_even = q[even_idx];
//         float q_odd = q[odd_idx];
//         float k_even = k[even_idx];
//         float k_odd = k[odd_idx];
        
//         q[even_idx] = q_even * cos_val - q_odd * sin_val;
//         q[odd_idx] = q_even * sin_val + q_odd * cos_val;
//         k[even_idx] = k_even * cos_val - k_odd * sin_val;
//         k[odd_idx] = k_even * sin_val + k_odd * cos_val;
//     }
// }



// RoPE kernel - Fixed to match CPU implementation exactly
__global__ void rope_kernel(float* q, float* k, int batch_size, int seq_len, int n_heads, int head_dim, int pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_dim = n_heads * head_dim;
    
    // Process pairs of elements (i, i+1) for rotation
    if (idx * 2 + 1 < total_dim) {
        int i = idx * 2;  // Start at even index
        
        // Calculate head dimension index (matches CPU: i % head_size)
        int head_size = head_dim;  // head_size = dim / n_heads, head_dim = head_size
        int head_dim_idx = i % head_size;
        
        // Calculate rotation frequency (matches CPU exactly)
        float freq = 1.0f / powf(10000.0f, head_dim_idx / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        // Determine how many vectors to rotate (matches CPU logic)
        int kv_dim = total_dim;  // Assuming n_kv_heads = n_heads for this test
        int rotn = i < kv_dim ? 2 : 1;
        
        // Apply rotation to query (and key if applicable) - matches CPU exactly
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? q : k;
            if (vec) {
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
}



// ----------------------------------------------------------------------------
// CUDA wrapper functions for Flash Attention

// Fix 3: Corrected CUDA wrapper
void cuda_flash_attention_fixed(
    float* d_output,
    float* d_query, 
    float* d_key_cache,
    float* d_value_cache,
    int seq_len, 
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float scale
) {
#if USE_CUDA
    dim3 gridDim(n_heads);
    dim3 blockDim(head_dim);
    
    // Shared memory: scores array + max_score + sum_exp
    size_t shared_mem_size = (pos + 3) * sizeof(float);

    
    flash_attention_kernel_fixed<<<gridDim, blockDim, shared_mem_size>>>(
        d_output, d_query, d_key_cache, d_value_cache,
        seq_len, n_heads, n_kv_heads, head_dim, pos, scale
    );

    // cudaStream_t stream = 0;

    // // Calculate optimal block size based on head_dim and shared memory constraints
    // int threads_per_block = min(1024, ((head_dim + 31) / 32) * 32); // Round up to warp size
    // int num_warps = (threads_per_block + 31) / 32;
    
    // // Calculate shared memory requirements
    // size_t smem_scores = (pos + 3) * sizeof(float);
    // size_t smem_query = head_dim * sizeof(float);
    // size_t smem_reductions = num_warps * sizeof(float);
    // size_t total_smem = smem_scores + smem_query + smem_reductions;
    
    // // Launch kernel
    // flash_attention_kernel_fixed<<<n_heads, threads_per_block, total_smem, stream>>>(
    //     d_output, d_query, d_key_cache, d_value_cache,
    //     seq_len, n_heads, n_kv_heads, head_dim, pos, scale
    // );
    
    CUDA_CHECK(cudaDeviceSynchronize());

#endif
}

void cuda_rmsnorm(float* d_output, float* d_input, float* d_weight, int size, int layer, int pos, int print_debug) {
    // Note: 'layer' is not used in this simplified version, but could be used for layer-specific weights
    // 'pos' is used to print debug information if needed
#if USE_CUDA
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    
    rmsnorm_kernel<<<gridDim, blockDim>>>(d_output, d_input, d_weight, size, 1e-5f, layer, pos, print_debug);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

void cuda_softmax(float* d_output, float* d_input, int size) {
#if USE_CUDA
    dim3 blockDim(256);
    dim3 gridDim(1);  // Single block for softmax
    
    softmax_kernel<<<gridDim, blockDim>>>(d_output, d_input, size);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}


// ----------------------------------------------------------------------------
// CPU implementations for fallback

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // This function computes xout = w^T * x
    // where w is stored in column-major format (as expected by cuBLAS)
    // w has dimensions (n x d), so w^T has dimensions (d x n)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            // For column-major storage: w[j * d + i] gives w[j][i]
            // Since we want w^T, we access w[i][j] which is w[j * d + i] 
            val += w[j * d + i] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Memory management and initialization

void malloc_run_state(RunState* s, Config* p, bool cuda_available) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    memset(s, 0, sizeof(RunState));
    
    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->k_original = (float *)calloc(kv_dim, sizeof(float));
    s->v_original = (float *)calloc(kv_dim, sizeof(float));
    s->k = s->k_original;
    s->v = s->v_original;
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || 
        !s->k_original || !s->v_original || !s->key_cache || !s->value_cache || 
        !s->att || !s->logits) {
        fprintf(stderr, "Host malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    
#if USE_CUDA
    if (cuda_available) {
        CUDA_CHECK(cudaMalloc(&s->d_x, p->dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_xb, p->dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_xb2, p->dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_hb, p->hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_hb2, p->hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_q, p->dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_k, kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_v, kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_att, p->n_heads * p->seq_len * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_logits, p->vocab_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_output, p->dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s->d_temp_storage, p->dim * sizeof(float)));
    }
#endif
}

void free_run_state(RunState* s) {
    if (s->x) { free(s->x); s->x = NULL; }
    if (s->xb) { free(s->xb); s->xb = NULL; }
    if (s->xb2) { free(s->xb2); s->xb2 = NULL; }
    if (s->hb) { free(s->hb); s->hb = NULL; }
    if (s->hb2) { free(s->hb2); s->hb2 = NULL; }
    if (s->q) { free(s->q); s->q = NULL; }
    if (s->att) { free(s->att); s->att = NULL; }
    if (s->logits) { free(s->logits); s->logits = NULL; }
    if (s->k_original) { free(s->k_original); s->k_original = NULL; }
    if (s->v_original) { free(s->v_original); s->v_original = NULL; }
    if (s->key_cache) { free(s->key_cache); s->key_cache = NULL; }
    if (s->value_cache) { free(s->value_cache); s->value_cache = NULL; }
    s->k = NULL; s->v = NULL;
    
#if USE_CUDA
    if (s->d_x) { cudaFree(s->d_x); s->d_x = NULL; }
    if (s->d_xb) { cudaFree(s->d_xb); s->d_xb = NULL; }
    if (s->d_xb2) { cudaFree(s->d_xb2); s->d_xb2 = NULL; }
    if (s->d_hb) { cudaFree(s->d_hb); s->d_hb = NULL; }
    if (s->d_hb2) { cudaFree(s->d_hb2); s->d_hb2 = NULL; }
    if (s->d_q) { cudaFree(s->d_q); s->d_q = NULL; }
    if (s->d_k) { cudaFree(s->d_k); s->d_k = NULL; }
    if (s->d_v) { cudaFree(s->d_v); s->d_v = NULL; }
    if (s->d_att) { cudaFree(s->d_att); s->d_att = NULL; }
    if (s->d_logits) { cudaFree(s->d_logits); s->d_logits = NULL; }
    if (s->d_key_cache) { cudaFree(s->d_key_cache); s->d_key_cache = NULL; }
    if (s->d_value_cache) { cudaFree(s->d_value_cache); s->d_value_cache = NULL; }
    if (s->d_output) { cudaFree(s->d_output); s->d_output = NULL; }
    if (s->d_temp_storage) { cudaFree(s->d_temp_storage); s->d_temp_storage = NULL; }
#endif
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    // Calculate key dimensions for tensor shapes
    int head_size = p->dim / p->n_heads;                    // Size of each attention head
    // int kv_head_size = p->dim * p->n_kv_heads / p->n_heads; // Size for key/value heads (for multi-query attention)
    unsigned long long n_layers = p->n_layers;
    
    printf("\n=== MEMORY MAPPING TRANSFORMER WEIGHTS ===\n");
    printf("Model Configuration:\n");
    printf("  - Vocabulary size: %d tokens\n", p->vocab_size);
    printf("  - Model dimension: %d\n", p->dim);
    printf("  - Number of layers: %d\n", p->n_layers);
    printf("  - Number of attention heads: %d\n", p->n_heads);
    printf("  - Number of key/value heads: %d\n", p->n_kv_heads);
    printf("  - Head size: %d\n", head_size);
    printf("  - Hidden dimension: %d\n", p->hidden_dim);
    printf("  - Max sequence length: %d\n", p->seq_len);
    printf("  - Shared classifier weights: %s\n\n", shared_weights ? "Yes" : "No");
    
    // 1. TOKEN EMBEDDING TABLE
    // Dimensions: [vocab_size, dim] - Maps each vocabulary token to a dense vector representation
    // Memory: vocab_size * dim * sizeof(float) bytes
    w->token_embedding_table = ptr;
    size_t token_embed_size = p->vocab_size * p->dim;
    printf("1. Token Embedding Table:\n");
    printf("   - Dimensions: [%d, %d] (vocab_size × dim)\n", p->vocab_size, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", token_embed_size, token_embed_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Maps vocabulary tokens to dense vector embeddings\n");
    ptr += token_embed_size;
    
    // 2. ATTENTION RMSNORM WEIGHTS  
    // Dimensions: [n_layers, dim] - RMSNorm scaling parameters for attention layers
    // Memory: n_layers * dim * sizeof(float) bytes
    w->rms_att_weight = ptr;
    size_t rms_att_size = p->n_layers * p->dim;
    printf("\n2. Attention RMSNorm Weights:\n");
    printf("   - Dimensions: [%d, %d] (n_layers × dim)\n", p->n_layers, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", rms_att_size, rms_att_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Scaling parameters for RMSNorm before self-attention\n");
    ptr += rms_att_size;
    
    // 3. QUERY PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, dim] - Projects input to query vectors for all attention heads
    // Memory: n_layers * dim * dim * sizeof(float) bytes
    w->wq = ptr;
    // size_t wq_size = n_layers * p->dim * p->dim;
    size_t wq_size = p->n_layers * p->dim * (p->n_heads * head_size);
    printf("\n3. Query Projection Weights (Wq):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × dim)\n", p->n_layers, p->dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", wq_size, wq_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to query vectors for multi-head attention\n");
    ptr += wq_size;
    
    // 4. KEY PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, kv_head_size] - Projects input to key vectors
    // For multi-query attention, kv_head_size may be smaller than dim
    w->wk = ptr;
    // size_t wk_size = n_layers * p->dim * kv_head_size;
    size_t wk_size = n_layers * p->dim * (p->n_kv_heads * head_size);
    printf("\n4. Key Projection Weights (Wk):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim ×  head_size)\n", p->n_layers, p->dim, head_size);
    printf("   - Memory: %zu floats = %.2f MB\n", wk_size, wk_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to key vectors for attention computation\n");
    ptr += wk_size;
    
    // 5. VALUE PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, kv_head_size] - Projects input to value vectors
    // Symmetric to key projection for multi-query attention
    w->wv = ptr;
    // size_t wv_size = n_layers * p->dim * kv_head_size;
    size_t wv_size = p->n_layers * p->dim * (p->n_kv_heads * head_size);
    printf("\n5. Value Projection Weights (Wv):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × kv_head_size)\n", p->n_layers, p->dim, head_size);
    printf("   - Memory: %zu floats = %.2f MB\n", wv_size, wv_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to value vectors for attention computation\n");
    ptr += wv_size;
    
    // 6. OUTPUT PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, dim] - Projects concatenated attention heads back to model dimension
    // Memory: n_layers * dim * dim * sizeof(float) bytes
    w->wo = ptr;
    // size_t wo_size = n_layers * p->dim * p->dim;
    size_t wo_size = p->n_layers * (p->n_heads * head_size) * p->dim;
    printf("\n6. Output Projection Weights (Wo):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × dim)\n", p->n_layers, p->dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", wo_size, wo_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects concatenated attention head outputs back to model dimension\n");
    ptr += wo_size;
    
    // 7. FEEDFORWARD RMSNORM WEIGHTS
    // Dimensions: [n_layers, dim] - RMSNorm scaling parameters for feedforward layers
    // Memory: n_layers * dim * sizeof(float) bytes
    w->rms_ffn_weight = ptr;
    size_t rms_ffn_size = n_layers * p->dim;
    printf("\n7. Feedforward RMSNorm Weights:\n");
    printf("   - Dimensions: [%d, %d] (n_layers × dim)\n", p->n_layers, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", rms_ffn_size, rms_ffn_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Scaling parameters for RMSNorm before feedforward network\n");
    ptr += rms_ffn_size;
    
    // 8. FEEDFORWARD GATE PROJECTION WEIGHTS (W1)
    // Dimensions: [n_layers, dim, hidden_dim] - First linear layer in SwiGLU feedforward
    // Memory: n_layers * dim * hidden_dim * sizeof(float) bytes
    w->w1 = ptr;
    size_t w1_size = n_layers * p->dim * p->hidden_dim;
    printf("\n8. Feedforward Gate Projection Weights (W1):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × hidden_dim)\n", p->n_layers, p->dim, p->hidden_dim);
    printf("   - Memory: %zu floats = %.2f MB\n", w1_size, w1_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Gate projection in SwiGLU activation (part of feedforward network)\n");
    ptr += w1_size;
    
    // 9. FEEDFORWARD DOWN PROJECTION WEIGHTS (W2)
    // Dimensions: [n_layers, hidden_dim, dim] - Final linear layer in feedforward network
    // Memory: n_layers * hidden_dim * dim * sizeof(float) bytes
    w->w2 = ptr;
    size_t w2_size = n_layers * p->hidden_dim * p->dim;
    printf("\n9. Feedforward Down Projection Weights (W2):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × hidden_dim × dim)\n", p->n_layers, p->hidden_dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", w2_size, w2_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects from hidden dimension back to model dimension in feedforward\n");
    ptr += w2_size;
    
    // 10. FEEDFORWARD UP PROJECTION WEIGHTS (W3)
    // Dimensions: [n_layers, dim, hidden_dim] - Second linear layer in SwiGLU feedforward
    // Memory: n_layers * dim * hidden_dim * sizeof(float) bytes
    w->w3 = ptr;
    size_t w3_size = n_layers * p->dim * p->hidden_dim;
    printf("\n10. Feedforward Up Projection Weights (W3):\n");
    printf("    - Dimensions: [%d, %d, %d] (n_layers × dim × hidden_dim)\n", p->n_layers, p->dim, p->hidden_dim);
    printf("    - Memory: %zu floats = %.2f MB\n", w3_size, w3_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("    - Purpose: Value projection in SwiGLU activation (multiplied with gated W1 output)\n");
    ptr += w3_size;
    
    // 11. FINAL RMSNORM WEIGHTS
    // Dimensions: [dim] - Final layer normalization before classifier
    // Memory: dim * sizeof(float) bytes
    w->rms_final_weight = ptr;
    size_t rms_final_size = p->dim;
    printf("\n11. Final RMSNorm Weights:\n");
    printf("    - Dimensions: [%d] (dim)\n", p->dim);
    printf("    - Memory: %zu floats = %.2f MB\n", rms_final_size, rms_final_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("    - Purpose: Final layer normalization before classification layer\n");
    ptr += rms_final_size;
    
    // Skip frequency caches (used for RoPE positional encoding, pre-computed)
    size_t freq_cache_size = p->seq_len * head_size / 2;
    printf("\n12. Skipping RoPE Frequency Caches:\n");
    printf("    - Cos cache size: %zu floats\n", freq_cache_size);
    printf("    - Sin cache size: %zu floats\n", freq_cache_size);
    printf("    - Purpose: Pre-computed cosine/sine values for Rotary Position Embedding\n");
    ptr += freq_cache_size; // cos frequencies
    ptr += freq_cache_size; // sin frequencies
    
    // 13. CLASSIFIER WEIGHTS (OPTIONAL)
    // Dimensions: [vocab_size, dim] - Final classification layer to predict next token
    // Memory: vocab_size * dim * sizeof(float) bytes (if not shared with embeddings)
    if (shared_weights) {
        w->wcls = w->token_embedding_table;  // Share weights with token embeddings (common practice)
        printf("\n13. Classifier Weights: SHARED with token embeddings\n");
        printf("    - Dimensions: [%d, %d] (vocab_size × dim)\n", p->vocab_size, p->dim);
        printf("    - Memory: 0 MB (shared with token embedding table)\n");
        printf("    - Purpose: Projects final hidden states to vocabulary logits (weight tying)\n");
    } else {
        w->wcls = ptr;  // Separate classifier weights
        size_t wcls_size = p->vocab_size * p->dim;
        printf("\n13. Classifier Weights: SEPARATE\n");
        printf("    - Dimensions: [%d, %d] (vocab_size × dim)\n", p->vocab_size, p->dim);
        printf("    - Memory: %zu floats = %.2f MB\n", wcls_size, wcls_size * sizeof(float) / (1024.0f * 1024.0f));
        printf("    - Purpose: Projects final hidden states to vocabulary logits\n");
    }
    
    // Calculate total memory usage
    size_t total_weights = token_embed_size + rms_att_size + wq_size + wk_size + wv_size + wo_size + 
                          rms_ffn_size + w1_size + w2_size + w3_size + rms_final_size;
    if (!shared_weights) {
        total_weights += p->vocab_size * p->dim;  // Add classifier weights if not shared
    }
    
    printf("\n=== TOTAL MEMORY SUMMARY ===\n");
    printf("Total transformer weights: %zu floats = %.2f MB\n", 
           total_weights, total_weights * sizeof(float) / (1024.0f * 1024.0f));
    printf("Frequency caches: %zu floats = %.2f MB\n", 
           2 * freq_cache_size, 2 * freq_cache_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("=====================================\n\n");
    
    // Initialize device weight pointers to NULL (will be allocated later if using GPU)
    memset(&w->d_token_embedding_table, 0, 11 * sizeof(float*));
}


void cuda_copy_weights_to_device(TransformerWeights* w, Config* p, bool cuda_available) {
#if USE_CUDA
    if (!cuda_available) return;
    
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    
    printf("Copying weights to GPU...\n");
    
    CUDA_CHECK(cudaMalloc(&w->d_token_embedding_table, p->vocab_size * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_token_embedding_table, w->token_embedding_table, 
                         p->vocab_size * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_rms_att_weight, n_layers * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_rms_att_weight, w->rms_att_weight, 
                         n_layers * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_wq, n_layers * p->dim * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_wq, w->wq, 
                         n_layers * p->dim * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_wk, n_layers * p->dim * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_wk, w->wk, 
                         n_layers * p->dim * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_wv, n_layers * p->dim * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_wv, w->wv, 
                         n_layers * p->dim * kv_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_wo, n_layers * p->dim * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_wo, w->wo, 
                         n_layers * p->dim * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_rms_ffn_weight, n_layers * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_rms_ffn_weight, w->rms_ffn_weight, 
                         n_layers * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_w1, n_layers * p->dim * p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_w1, w->w1, 
                         n_layers * p->dim * p->hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_w2, n_layers * p->hidden_dim * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_w2, w->w2, 
                         n_layers * p->hidden_dim * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_w3, n_layers * p->dim * p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_w3, w->w3, 
                         n_layers * p->dim * p->hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_rms_final_weight, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_rms_final_weight, w->rms_final_weight, 
                         p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&w->d_wcls, p->vocab_size * p->dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w->d_wcls, w->wcls, 
                         p->vocab_size * p->dim * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Weights copied to GPU successfully.\n");
#endif
}

void free_cuda_weights(TransformerWeights* w) {
#if USE_CUDA
    if (w->d_token_embedding_table) { cudaFree(w->d_token_embedding_table); w->d_token_embedding_table = NULL; }
    if (w->d_rms_att_weight) { cudaFree(w->d_rms_att_weight); w->d_rms_att_weight = NULL; }
    if (w->d_rms_ffn_weight) { cudaFree(w->d_rms_ffn_weight); w->d_rms_ffn_weight = NULL; }
    if (w->d_wq) { cudaFree(w->d_wq); w->d_wq = NULL; }
    if (w->d_wk) { cudaFree(w->d_wk); w->d_wk = NULL; }
    if (w->d_wv) { cudaFree(w->d_wv); w->d_wv = NULL; }
    if (w->d_wo) { cudaFree(w->d_wo); w->d_wo = NULL; }
    if (w->d_w1) { cudaFree(w->d_w1); w->d_w1 = NULL; }
    if (w->d_w2) { cudaFree(w->d_w2); w->d_w2 = NULL; }
    if (w->d_w3) { cudaFree(w->d_w3); w->d_w3 = NULL; }
    if (w->d_rms_final_weight) { cudaFree(w->d_rms_final_weight); w->d_rms_final_weight = NULL; }
    if (w->d_wcls) { cudaFree(w->d_wcls); w->d_wcls = NULL; }
#endif
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path, bool cuda_available) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config, cuda_available);
    cuda_copy_weights_to_device(&t->weights, &t->config, cuda_available);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    
    free_cuda_weights(&t->weights);
    free_run_state(&t->state);
}


// ----------------------------------------------------------------------------
// Tokenizer implementation

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    printf("\n=== BUILDING TOKENIZER ===\n");
    printf("Tokenizer file: %s\n", tokenizer_path);
    printf("Expected vocabulary size: %d tokens\n\n", vocab_size);
    
    // Initialize tokenizer structure with vocabulary size
    // vocab_size: Total number of tokens in the vocabulary (typically 256 for byte-level BPE)
    t->vocab_size = vocab_size;
    printf("1. Setting vocabulary size: %d\n", vocab_size);
    
    // Allocate memory for vocabulary array (array of string pointers)
    // Dimensions: [vocab_size] pointers to char* strings
    // Memory: vocab_size * sizeof(char*) bytes for pointer array
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    printf("2. Allocated vocabulary pointer array: %d pointers = %zu bytes\n", 
           vocab_size, vocab_size * sizeof(char*));
    
    // Allocate memory for vocabulary scores (BPE merge priorities)
    // Dimensions: [vocab_size] float values
    // Memory: vocab_size * sizeof(float) bytes
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    printf("3. Allocated vocabulary scores array: %d floats = %zu bytes\n", 
           vocab_size, vocab_size * sizeof(float));
    
    // Initialize sorted vocabulary as NULL (will be created lazily when needed)
    // Purpose: Sorted version of vocabulary for fast binary search during encoding
    t->sorted_vocab = NULL;
    printf("4. Initialized sorted vocabulary as NULL (created lazily)\n");
    
    // Initialize byte pieces array for single-byte token representations
    // Dimensions: [256][2] - 256 single bytes, each represented as 2-char string (byte + null terminator)
    // Memory: 512 bytes (256 * 2 chars) - stores individual byte values as strings
    printf("5. Initializing byte pieces for single-byte tokens:\n");
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;      // Store the actual byte value
        t->byte_pieces[i * 2 + 1] = '\0';              // Null terminator
    }
    printf("   - Created 256 single-byte token representations (512 bytes total)\n");
    printf("   - Examples: byte_pieces[0] = '\\0', byte_pieces[65*2] = 'A', byte_pieces[32*2] = ' '\n");
    
    // Open tokenizer binary file for reading
    printf("\n6. Opening tokenizer file: %s\n", tokenizer_path);
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { 
        fprintf(stderr, "Error: couldn't load tokenizer file %s\n", tokenizer_path); 
        exit(EXIT_FAILURE); 
    }
    
    // Read maximum token length from file header
    // max_token_length: Maximum length of any token string in the vocabulary
    // Used for buffer allocation during encoding/decoding
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "Error: failed to read max_token_length\n"); 
        exit(EXIT_FAILURE); 
    }
    printf("7. Read max token length: %d characters\n", t->max_token_length);
    
    // Read vocabulary entries from file
    // File format: For each token: [score:float][length:int][string:char[length]]
    printf("\n8. Reading vocabulary entries from file:\n");
    int len;
    size_t total_vocab_string_memory = 0;
    int num_examples_to_show = 20;  // Show first 20 tokens as examples
    
    for (int i = 0; i < vocab_size; i++) {
        // Read BPE merge score for this token (higher = merged later in BPE algorithm)
        // Dimensions: 1 float per token
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { 
            fprintf(stderr, "Error: failed to read vocab score for token %d\n", i); 
            exit(EXIT_FAILURE);
        }
        
        // Read length of token string
        // Dimensions: 1 int per token (string length)
        if (fread(&len, sizeof(int), 1, file) != 1) { 
            fprintf(stderr, "Error: failed to read token length for token %d\n", i); 
            exit(EXIT_FAILURE); 
        }
        
        // Allocate memory for token string (including null terminator)
        // Dimensions: [len + 1] characters per token
        // Memory: (len + 1) bytes per token string
        t->vocab[i] = (char *)malloc(len + 1);
        total_vocab_string_memory += (len + 1);
        
        // Read token string from file
        if (fread(t->vocab[i], len, 1, file) != 1) { 
            fprintf(stderr, "Error: failed to read vocab string for token %d\n", i); 
            exit(EXIT_FAILURE); 
        }
        t->vocab[i][len] = '\0';  // Add null terminator
        
        // Print examples of vocabulary tokens (first few tokens)
        if (i < num_examples_to_show) {
            printf("   Token %3d: score=%8.4f, length=%2d, string=\"", i, t->vocab_scores[i], len);
            // Print token string safely (handle special characters)
            for (int j = 0; j < len; j++) {
                char c = t->vocab[i][j];
                if (c >= 32 && c <= 126) {  // Printable ASCII
                    printf("%c", c);
                } else if (c == '\n') {
                    printf("\\n");
                } else if (c == '\t') {
                    printf("\\t");
                } else if (c == ' ') {
                    printf("·");  // Visible space
                } else {
                    printf("\\x%02x", (unsigned char)c);  // Hex representation
                }
            }
            printf("\"\n");
        } else if (i == num_examples_to_show) {
            printf("   ... (showing first %d tokens only)\n", num_examples_to_show);
        }
    }
    
    fclose(file);
    
    // Calculate and display memory usage summary
    printf("\n=== TOKENIZER MEMORY SUMMARY ===\n");
    printf("Vocabulary pointer array: %zu bytes\n", vocab_size * sizeof(char*));
    printf("Vocabulary scores array: %zu bytes\n", vocab_size * sizeof(float));
    printf("Vocabulary strings total: %zu bytes\n", total_vocab_string_memory);
    printf("Byte pieces array: 512 bytes\n");
    printf("Sorted vocabulary: 0 bytes (created lazily)\n");
    
    size_t total_memory = vocab_size * sizeof(char*) + vocab_size * sizeof(float) + 
                         total_vocab_string_memory + 512;
    printf("Total tokenizer memory: %zu bytes = %.2f KB\n", 
           total_memory, total_memory / 1024.0f);
    
    // Display tokenizer statistics
    printf("\n=== TOKENIZER STATISTICS ===\n");
    printf("Total vocabulary size: %d tokens\n", vocab_size);
    printf("Maximum token length: %d characters\n", t->max_token_length);
    
    // Analyze vocabulary composition
    int single_char_tokens = 0;
    int multi_char_tokens = 0;
    int special_tokens = 0;
    float min_score = t->vocab_scores[0], max_score = t->vocab_scores[0];
    
    for (int i = 0; i < vocab_size; i++) {
        int token_len = strlen(t->vocab[i]);
        if (token_len == 1) {
            single_char_tokens++;
        } else {
            multi_char_tokens++;
        }
        
        // Check for special tokens (usually start with < and end with >)
        if (t->vocab[i][0] == '<' && t->vocab[i][token_len-1] == '>') {
            special_tokens++;
        }
        
        // Track score range
        if (t->vocab_scores[i] < min_score) min_score = t->vocab_scores[i];
        if (t->vocab_scores[i] > max_score) max_score = t->vocab_scores[i];
    }
    
    printf("Token composition:\n");
    printf("  - Single character tokens: %d\n", single_char_tokens);
    printf("  - Multi-character tokens: %d\n", multi_char_tokens);
    printf("  - Special tokens (< >): %d\n", special_tokens);
    printf("Score range: %.4f to %.4f\n", min_score, max_score);
    
    // Show some interesting token examples
    printf("\nExample token categories:\n");
    printf("  - BOS (Beginning of Sequence): token 1 = \"%s\"\n", 
           vocab_size > 1 ? t->vocab[1] : "N/A");
    printf("  - EOS (End of Sequence): token 2 = \"%s\"\n", 
           vocab_size > 2 ? t->vocab[2] : "N/A");
    
    // Show some common byte tokens
    if (vocab_size >= 256) {
        printf("  - Space character: token %d = \"%s\"\n", 32, t->vocab[32]);
        printf("  - Letter 'A': token %d = \"%s\"\n", 65, t->vocab[65]);
        printf("  - Letter 'a': token %d = \"%s\"\n", 97, t->vocab[97]);
    }
    
    printf("===============================\n\n");
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok;
    tok.str = (char *)str;
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    printf("\n=== ENCODING TEXT TO TOKENS ===\n");
    printf("Input text: \"%s\"\n", text ? text : "NULL");
    printf("Add BOS (Beginning of Sequence): %s\n", bos ? "Yes" : "No");
    printf("Add EOS (End of Sequence): %s\n", eos ? "Yes" : "No");
    printf("Text length: %zu characters\n\n", text ? strlen(text) : 0);
    
    // Input validation - ensure we have valid text to encode
    if (text == NULL) { 
        fprintf(stderr, "Error: cannot encode NULL text\n"); 
        exit(EXIT_FAILURE); 
    }

    // STEP 1: Initialize or create sorted vocabulary for fast lookups
    // sorted_vocab: Array of TokenIndex structs sorted alphabetically by string
    // Dimensions: [vocab_size] TokenIndex structs
    // Memory: vocab_size * sizeof(TokenIndex) bytes
    printf("1. Setting up sorted vocabulary for binary search:\n");
    if (t->sorted_vocab == NULL) {
        printf("   - Creating sorted vocabulary array...\n");
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        printf("   - Memory allocated: %d TokenIndex structs = %zu bytes\n", 
               t->vocab_size, t->vocab_size * sizeof(TokenIndex));
        
        // Copy vocabulary strings and IDs into sortable structure
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];  // Pointer to vocabulary string
            t->sorted_vocab[i].id = i;             // Original token ID
        }
        
        // Sort alphabetically for binary search (O(log n) lookups)
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
        printf("   - Sorted %d vocabulary entries alphabetically\n", t->vocab_size);
    } else {
        printf("   - Using existing sorted vocabulary\n");
    }

    // STEP 2: Allocate working buffer for token construction
    // str_buffer: Temporary buffer for building token strings during UTF-8 processing
    // Dimensions: [max_token_length*2 + 3] characters  
    // Memory: (max_token_length*2 + 3) bytes
    // Extra space for: UTF-8 sequences, concatenations, safety margin
    char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t buffer_size = t->max_token_length*2 +1 +2;
    printf("\n2. Allocated working buffer: %zu bytes\n", buffer_size);
    
    // Initialize encoding state
    size_t str_len = 0;      // Current length of string being built
    *n_tokens = 0;          // Reset output token count
    printf("3. Initialized encoding state: str_len=0, n_tokens=0\n");

    // STEP 3: Add Beginning of Sequence token if requested
    // BOS token (typically ID=1) marks the start of a sequence
    printf("\n4. Processing special tokens:\n");
    if (bos) {
        tokens[(*n_tokens)++] = 1;  // BOS token ID is conventionally 1
        printf("   - Added BOS token: ID=1 at position %d\n", *n_tokens - 1);
    }

    // STEP 4: Add dummy space prefix for proper tokenization
    // Many tokenizers expect a space at the beginning to handle word boundaries correctly
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix != -1) {
            tokens[(*n_tokens)++] = dummy_prefix;
            printf("   - Added space prefix token: ID=%d at position %d\n", dummy_prefix, *n_tokens - 1);
        } else {
            printf("   - Warning: Space token not found in vocabulary\n");
        }
    }

    printf("\n5. Initial token array state:\n");
    printf("   Current tokens: [");
    for (int i = 0; i < *n_tokens; i++) {
        printf("%d%s", tokens[i], (i < *n_tokens - 1) ? ", " : "");
    }
    printf("]\n");
    printf("   Token count: %d\n", *n_tokens);

    // STEP 5: Process input text character by character (UTF-8 aware)
    printf("\n6. Processing input text character by character:\n");
    int char_count = 0;
    
    for (char *c = text; *c != '\0'; c++) {
        char_count++;
        printf("   Character %d: '%c' (0x%02X)\n", char_count, 
               (*c >= 32 && *c <= 126) ? *c : '?', (unsigned char)*c);
        
        // Check if this is a UTF-8 continuation byte (starts with bits 10)
        // If not, reset string length (start of new UTF-8 sequence)
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
            printf("     -> UTF-8 sequence start (or ASCII)\n");
        } else {
            printf("     -> UTF-8 continuation byte\n");
        }
        
        // Add character to current string buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        printf("     -> str_buffer: \"%s\" (length=%zu)\n", str_buffer, str_len);
        
        // Continue if next character is UTF-8 continuation and we haven't exceeded limit
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            printf("     -> Continuing UTF-8 sequence...\n");
            continue;
        }
        
        // Try to find complete UTF-8 character/sequence in vocabulary
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        printf("     -> Vocabulary lookup for \"%s\": ", str_buffer);
        
        if (id != -1) {
            // Found in vocabulary - add as single token
            tokens[(*n_tokens)++] = id;
            printf("FOUND (ID=%d)\n", id);
            printf("     -> Added token %d at position %d\n", id, *n_tokens - 1);
        } else {
            // Not found - encode as individual bytes (with offset +3)
            printf("NOT FOUND - encoding as bytes\n");
            for (int i = 0; i < str_len; i++) {
                int byte_token = (unsigned char)str_buffer[i] + 3;
                tokens[(*n_tokens)++] = byte_token;
                printf("     -> Byte '%c' (0x%02X) -> token %d at position %d\n", 
                       str_buffer[i], (unsigned char)str_buffer[i], byte_token, *n_tokens - 1);
            }
        }
        str_len = 0;  // Reset for next character/sequence
        
        printf("     -> Current token count: %d\n", *n_tokens);
    }

    printf("\n7. After initial tokenization:\n");
    printf("   Tokens: [");
    for (int i = 0; i < *n_tokens; i++) {
        printf("%d%s", tokens[i], (i < *n_tokens - 1) ? ", " : "");
    }
    printf("]\n");
    printf("   Token count: %d\n", *n_tokens);
    printf("   Token strings: [");
    for (int i = 0; i < *n_tokens; i++) {
        printf("\"%s\"%s", t->vocab[tokens[i]], (i < *n_tokens - 1) ? ", " : "");
    }
    printf("]\n");

    // STEP 6: BPE (Byte Pair Encoding) merge phase
    // Iteratively find and merge the highest-scoring adjacent token pairs
    printf("\n8. Starting BPE merge phase:\n");
    int merge_iteration = 0;
    
    while (1) {
        merge_iteration++;
        printf("   Merge iteration %d:\n", merge_iteration);
        
        // Find the best pair to merge (highest vocabulary score)
        float best_score = -1e10;  // Start with very low score
        int best_id = -1;           // Token ID of merged pair
        int best_idx = -1;          // Position where merge should happen
        
        printf("     Searching for best merge candidate...\n");
        
        // Check all adjacent token pairs
        for (int i = 0; i < (*n_tokens - 1); i++) {
            // Concatenate two adjacent tokens to form candidate merge
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            
            // Look up concatenated string in vocabulary
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
                printf("     -> Better merge found: \"%s\"+\"%s\" = \"%s\" (ID=%d, score=%.4f) at position %d\n",
                       t->vocab[tokens[i]], t->vocab[tokens[i+1]], str_buffer, id, best_score, i);
            }
        }
        
        // If no merge found, we're done
        if (best_idx == -1) {
            printf("     -> No more merges possible. BPE complete.\n");
            break;
        }
        
        printf("     -> Applying merge: tokens[%d]+tokens[%d] -> token %d\n", 
               best_idx, best_idx + 1, best_id);
        printf("     -> Before merge: [");
        for (int i = 0; i < *n_tokens; i++) {
            printf("%d%s", tokens[i], (i < *n_tokens - 1) ? ", " : "");
        }
        printf("]\n");
        
        // Apply the merge: replace tokens[best_idx] and tokens[best_idx+1] with best_id
        tokens[best_idx] = best_id;
        
        // Shift remaining tokens left to fill the gap
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // Reduce token count by 1
        
        printf("     -> After merge:  [");
        for (int i = 0; i < *n_tokens; i++) {
            printf("%d%s", tokens[i], (i < *n_tokens - 1) ? ", " : "");
        }
        printf("]\n");
        printf("     -> New token count: %d\n", *n_tokens);
        
        // Safety check to prevent infinite loops
        if (merge_iteration > 1000) {
            printf("     -> WARNING: Stopping after 1000 merge iterations\n");
            break;
        }
    }

    // STEP 7: Add End of Sequence token if requested
    printf("\n9. Finalizing token sequence:\n");
    if (eos) {
        tokens[(*n_tokens)++] = 2;  // EOS token ID is conventionally 2
        printf("   - Added EOS token: ID=2 at position %d\n", *n_tokens - 1);
    }

    // STEP 8: Display final results
    printf("\n=== ENCODING COMPLETE ===\n");
    printf("Final token count: %d\n", *n_tokens);
    printf("Final token IDs: [");
    for (int i = 0; i < *n_tokens; i++) {
        printf("%d%s", tokens[i], (i < *n_tokens - 1) ? ", " : "");
    }
    printf("]\n");
    
    printf("Final token strings: [");
    for (int i = 0; i < *n_tokens; i++) {
        printf("\"%s\"%s", t->vocab[tokens[i]], (i < *n_tokens - 1) ? ", " : "");
    }
    printf("]\n");
    
    printf("Reconstructed text: \"");
    for (int i = 0; i < *n_tokens; i++) {
        // Skip special tokens for reconstruction
        // if (tokens[i] != 1 && tokens[i] != 2) {  // Skip BOS and EOS
            printf("%s", t->vocab[tokens[i]]);
        // }
    }
    printf("\"\n");
    
    // Memory usage summary
    printf("\nMemory usage during encoding:\n");
    printf("  - Working buffer: %zu bytes\n", buffer_size);
    printf("  - Output tokens array: %d * %zu = %zu bytes\n", 
           *n_tokens, sizeof(int), *n_tokens * sizeof(int));
    printf("  - Sorted vocabulary (if created): %zu bytes\n", 
           t->sorted_vocab ? t->vocab_size * sizeof(TokenIndex) : 0);
    
    printf("BPE merge iterations: %d\n", merge_iteration - 1);
    printf("==========================\n\n");
    
    // Clean up temporary buffer
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler implementation

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_probs(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probs);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex *) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

// Fix 5: Add debugging to the sampling function
int sample_debug(Sampler* sampler, float* logits) {
    // Check for problematic values
    float max_logit = -INFINITY, min_logit = INFINITY;
    int nan_count = 0;
    
    for (int i = 0; i < sampler->vocab_size; i++) {
        if (isnan(logits[i])) {
            nan_count++;
        } else {
            if (logits[i] > max_logit) max_logit = logits[i];
            if (logits[i] < min_logit) min_logit = logits[i];
        }
    }
    
    if (nan_count > 0) {
        printf("ERROR: %d NaN values in logits! Using fallback sampling.\n", nan_count);
        return 1; // Return a safe token
    }
    
    // printf("Logits range: [%.3f, %.3f], temp=%.3f\n", min_logit, max_logit, sampler->temperature);
    
    int next;
    if (sampler->temperature == 0.0f || sampler->temperature < 1e-6f) {
        next = sample_argmax(logits, sampler->vocab_size);
        printf("Using argmax sampling -> token %d\n", next);
    } else {
        // Apply temperature
        for (int q = 0; q < sampler->vocab_size; q++) { 
            logits[q] /= sampler->temperature; 
        }
        
        // Apply softmax
        softmax(logits, sampler->vocab_size);
        
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
        // printf("Using temperature sampling (coin=%.3f) -> token %d\n", coin, next);
    }
    
    return next;
}



// ----------------------------------------------------------------------------
// Time utilities

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// Flash Attention CPU implementation (reference)

// Fix 1: Correct Flash Attention CPU implementation with proper causal masking
void flash_attention_cpu_fixed(
    float* output,           // [n_heads, head_dim]
    float* query,            // [n_heads, head_dim]  
    float* key_cache,        // [seq_len, n_kv_heads, head_dim]
    float* value_cache,      // [seq_len, n_kv_heads, head_dim]
    int seq_len,
    int n_heads,
    int n_kv_heads, 
    int head_dim,
    int pos                  // Current position for causal masking
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Clear output
    memset(output, 0, n_heads * head_dim * sizeof(float));
    
    for (int h = 0; h < n_heads; h++) {
        int kv_head = h * n_kv_heads / n_heads; // For grouped-query attention
        
        float* q = query + h * head_dim;
        float* out = output + h * head_dim;
        
        // Find maximum score for numerical stability
        float max_score = -INFINITY;
        float* scores = (float*)malloc((pos + 1) * sizeof(float));  // Only compute up to current position (causal)
        
        for (int t = 0; t <= pos; t++) {
            float* k = key_cache + t * n_kv_heads * head_dim + kv_head * head_dim;
            
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * k[d];
            }
            score *= scale;
            scores[t] = score;
            
            if (score > max_score) max_score = score;
        }
        
        // Compute softmax with numerical stability
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        
        // Apply attention weights to values
        for (int t = 0; t <= pos; t++) {
            float* v = value_cache + t * n_kv_heads * head_dim + kv_head * head_dim;
            float weight = scores[t] / sum_exp;
            
            for (int d = 0; d < head_dim; d++) {
                out[d] += weight * v[d];
            }
        }
        
        free(scores);  // Free the dynamically allocated memory
    }
}


// ----------------------------------------------------------------------------
// Transformer forward pass

// Fix 4: Corrected forward function with proper tensor reshaping
float* forward_fixed(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // Copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));
    
    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        
        // Attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        
        // QKV matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        
        // RoPE relative positional encoding
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            
            for (int i = 0; i < head_size; i += 2) {
                if (i + 1 < head_size) {  // Ensure we don't go out of bounds
                    float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    
                    float q0 = q[i];
                    float q1 = q[i + 1];
                    q[i] = q0 * fcr - q1 * fci;
                    q[i + 1] = q0 * fci + q1 * fcr;
                }
            }
        }
        
        // Apply RoPE to keys as well
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* k = s->k + h * head_size;
            
            for (int i = 0; i < head_size; i += 2) {
                if (i + 1 < head_size) {
                    float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    
                    float k0 = k[i];
                    float k1 = k[i + 1];
                    k[i] = k0 * fcr - k1 * fci;
                    k[i + 1] = k0 * fci + k1 * fcr;
                }
            }
        }
        
        // Save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(float));
        
        // Flash Attention with proper parameters
        printf("Layer %d: Applying Flash Attention at position %d\n", l, pos);
        
        if (g_cuda_available) {
            // GPU Flash Attention (simplified for debugging)
            printf("Using CPU Flash Attention for debugging...\n");
            flash_attention_cpu_fixed(s->xb, s->q, s->key_cache + loff, s->value_cache + loff,
                                     p->seq_len, p->n_heads, p->n_kv_heads, head_size, pos);
        } else {
            // CPU Flash Attention
            flash_attention_cpu_fixed(s->xb, s->q, s->key_cache + loff, s->value_cache + loff,
                                     p->seq_len, p->n_heads, p->n_kv_heads, head_size, pos);
        }
        
        // Final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        
        // Residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        
        // FFN rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
        
        // FFN: w1 and w3 projections
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x)=1/(1+e^(-x))
            val = val / (1.0f + expf(-val));
            // elementwise multiply with w3(x)
            val = val * s->hb2[i];
            s->hb[i] = val;
        }
        
        // Final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);
        
        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    
    // Final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);
    
    // Classifier into logits
    matmul(s->logits, x, w->wcls, dim, p->vocab_size);
    
    // Debug: Print some logits
    if (pos < 5) {
        printf("Position %d logits sample: [", pos);
        for (int i = 0; i < 10; i++) {
            printf("%.3f ", s->logits[i]);
        }
        printf("...]\n");
        
        // Check for NaN or infinite values
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < p->vocab_size; i++) {
            if (isnan(s->logits[i])) nan_count++;
            if (isinf(s->logits[i])) inf_count++;
        }
        if (nan_count > 0 || inf_count > 0) {
            printf("WARNING: Found %d NaN and %d infinite values in logits!\n", nan_count, inf_count);
        }
    }
    
    return s->logits;
}

// CORRECTED forward function with proper Flash Attention integration

float* forward_gpu(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float scale = 1.0f / sqrtf((float)head_size);

#if USE_CUDA
    if (!g_cuda_available) {
        return forward_fixed(transformer, token, pos);  // Fall back to CPU
    }

    // Copy token embedding to GPU
    float* token_embedding = w->token_embedding_table + token * dim;
    CUDA_CHECK(cudaMemcpy(s->d_x, token_embedding, dim * sizeof(float), cudaMemcpyHostToDevice));

    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // printf("GPU Layer %d processing...\n", l);

        // Attention rmsnorm on GPU
        float* layer_rms_att_weight = w->d_rms_att_weight + l * dim;
        cuda_rmsnorm(s->d_xb, s->d_x, layer_rms_att_weight, dim, l, pos, 1);

        
        // QKV matmuls using cuBLAS
        float* layer_wq = w->d_wq + l * dim * dim;
        float* layer_wk = w->d_wk + l * dim * kv_dim;  
        float* layer_wv = w->d_wv + l * dim * kv_dim;
        
        // Q = xb * Wq^T
        const float alpha = 1.0f, beta = 0.0f;
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, layer_wq, dim, s->d_xb, 1, &beta, s->d_q, 1);

        // Q = xb * Wq^T (replaced cublasSgemv)
        cuda_sgemv(s->d_q, layer_wq, s->d_xb, dim, dim, alpha, beta, true);


        // K = xb * Wk^T  
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, kv_dim, &alpha, layer_wk, dim, s->d_xb, 1, &beta, s->d_k, 1);
        
        // K = xb * Wk^T (replaced cublasSgemv)
        cuda_sgemv(s->d_k, layer_wk, s->d_xb, dim, kv_dim, alpha, beta, true);


        // V = xb * Wv^T
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, kv_dim, &alpha, layer_wv, dim, s->d_xb, 1, &beta, s->d_v, 1);

        // V = xb * Wv^T (replaced cublasSgemv)
        cuda_sgemv(s->d_v, layer_wv, s->d_xb, dim, kv_dim, alpha, beta, true);       

        // Apply RoPE on GPU
        int total_elements = 1 * 1 * p->n_heads * head_size;
        dim3 blockDim(256);
        dim3 gridDim((total_elements / 2 + blockDim.x - 1) / blockDim.x);  // Process pairs
        
        rope_kernel<<<gridDim, blockDim>>>(s->d_q, s->d_k, 1, 1, p->n_heads, head_size, pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update KV cache on GPU
        int loff = l * p->seq_len * kv_dim;
        float* d_key_cache_row = s->d_key_cache + loff + pos * kv_dim;
        float* d_value_cache_row = s->d_value_cache + loff + pos * kv_dim;
        
        CUDA_CHECK(cudaMemcpy(d_key_cache_row, s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_value_cache_row, s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        // **HERE IS WHERE FLASH ATTENTION IS ACTUALLY USED**
        // printf("Applying Flash Attention on GPU for layer %d, position %d\n", l, pos);
        
        // Prepare Flash Attention inputs
        float* d_query_reshaped = s->d_q;  // [1, 1, n_heads, head_size]
        float* d_key_cache_layer = s->d_key_cache + loff;    // [1, seq_len, n_kv_heads, head_size]  
        float* d_value_cache_layer = s->d_value_cache + loff; // [1, seq_len, n_kv_heads, head_size]
        
        // Call our custom Flash Attention CUDA kernel
        cuda_flash_attention_fixed(
            s->d_xb,                    // output: [n_heads, head_size] 
            d_query_reshaped,           // query:  [n_heads, head_size]
            d_key_cache_layer,          // key:    [seq_len, n_kv_heads, head_size]
            d_value_cache_layer,        // value:  [seq_len, n_kv_heads, head_size] 
            pos + 1,                    // seq_len = current position + 1 (causal)
            p->n_heads,                 // n_heads
            p->n_kv_heads,              // n_kv_heads
            head_size,                  // head_dim
            pos,                        // pos (current position for causal masking)
            scale                       // attention scale
        );

        // Output projection: xb2 = xb * Wo^T
        float* layer_wo = w->d_wo + l * dim * dim;
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, layer_wo, dim, s->d_xb, 1, &beta, s->d_xb2, 1);
        cuda_sgemv(s->d_xb2, layer_wo, s->d_xb, dim, dim, alpha, beta, true);


        // Residual connection: x = x + xb2
        // cublasSaxpy(cublas_handle, dim, &alpha, s->d_xb2, 1, s->d_x, 1);
        cuda_saxpy(s->d_x, s->d_xb2, alpha, dim);

        // FFN rmsnorm
        float* layer_rms_ffn_weight = w->d_rms_ffn_weight + l * dim;
        cuda_rmsnorm(s->d_xb, s->d_x, layer_rms_ffn_weight, dim, l, pos, 0);

        // FFN: w1 and w3 projections
        float* layer_w1 = w->d_w1 + l * dim * hidden_dim;
        float* layer_w3 = w->d_w3 + l * dim * hidden_dim;
        
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, hidden_dim, &alpha, layer_w1, dim, s->d_xb, 1, &beta, s->d_hb, 1);
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, hidden_dim, &alpha, layer_w3, dim, s->d_xb, 1, &beta, s->d_hb2, 1);

        cuda_sgemv(s->d_hb, layer_w1, s->d_xb, dim, hidden_dim, alpha, beta, true);
        cuda_sgemv(s->d_hb2, layer_w3, s->d_xb, dim, hidden_dim, alpha, beta, true);

        // SwiGLU activation on GPU
        dim3 swiglu_grid((hidden_dim + 255) / 256);
        dim3 swiglu_block(256);
        swiglu_kernel<<<swiglu_grid, swiglu_block>>>(s->d_hb, s->d_hb, s->d_hb2, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Final FFN projection: xb = hb * W2^T
        float* layer_w2 = w->d_w2 + l * hidden_dim * dim;
        // cublasSgemv(cublas_handle, CUBLAS_OP_T, hidden_dim, dim, &alpha, layer_w2, hidden_dim, s->d_hb, 1, &beta, s->d_xb, 1);
        cuda_sgemv(s->d_xb, layer_w2, s->d_hb, hidden_dim, dim, alpha, beta, true);

        // Residual connection: x = x + xb  
        // cublasSaxpy(cublas_handle, dim, &alpha, s->d_xb, 1, s->d_x, 1);
        cuda_saxpy(s->d_x, s->d_xb, alpha, dim);
    }

    // Final rmsnorm
    cuda_rmsnorm(s->d_x, s->d_x, w->d_rms_final_weight, dim, 1000, pos, 0);

    // Classifier: logits = x * Wcls^T
    const float alpha = 1.0f, beta = 0.0f;
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, p->vocab_size, &alpha, w->d_wcls, dim, s->d_x, 1, &beta, s->d_logits, 1);
    cuda_sgemv(s->d_logits, w->d_wcls, s->d_x, dim, p->vocab_size, alpha, beta, true);

    // Copy logits back to host
    CUDA_CHECK(cudaMemcpy(s->logits, s->d_logits, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    return s->logits;
#else
    return forward_fixed(transformer, token, pos);  // Fall back to CPU
#endif
}


// ----------------------------------------------------------------------------
// Optimized GPU generation function
// CORRECTED generate_fused_optimized - actually calls the GPU functions

void generate_fused_optimized(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    printf("=== FUSED OPTIMIZED GPU GENERATION ===\n");
    printf("Using Flash Attention with CUDA acceleration\n");
    printf("Steps: %d\n", steps);
    printf("Prompt: \"%s\"\n", prompt ? prompt : "");
    printf("=====================================\n\n");
    
    // Encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt ? prompt : "") + 3) * sizeof(int));
    if (prompt) encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        prompt_tokens[0] = 1; // BOS
        num_prompt_tokens = 1;
    }
    
    long start = 0;
    int next;
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0; // position in the sequence
    
    printf("Starting generation with token %d (\"%s\")\n", token, tokenizer->vocab[token]);
    printf("Initial prompt tokens: %d\n", num_prompt_tokens);
    
    while (pos < steps) {
        
        // **THIS IS WHERE generate_fused_optimized IS DIFFERENT FROM generate**
        // Use GPU-accelerated forward pass with Flash Attention
        float* logits;
        if (g_cuda_available) {
            // if (pos < 10) {
            //     printf("Using GPU forward pass with Flash Attention...\n");
            // }
            logits = forward_gpu(transformer, token, pos);  // ← USES GPU + Flash Attention
        } else {
            printf("CUDA not available, falling back to CPU...\n");
            logits = forward_fixed(transformer, token, pos);      // ← Falls back to CPU
        }

        // if (pos < 10) {
        //     printf("\nPrompt: \"%s\"\n", prompt);
        //     printf("Initial token: %d (%s)\n", token, tokenizer->vocab[token]);

        //     printf("\n Logits : [ ");
        //     for(int i = 0; i < 10; i++) {
        //         printf("%.4f, ", logits[i]);
        //     }
        //     printf(" ]\n\n");

        // }
        
        // Advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // If we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
            // printf("Prompt[%d]: %d (\"%s\")", pos, next, tokenizer->vocab[next]);
        } else {
            // Otherwise sample the next token from the logits
            next = sample_debug(sampler, logits);
            // printf("Generated[%d]: %d (\"%s\")", pos, next, tokenizer->vocab[next]);
        }
        pos++;
        
        // Data-dependent terminating condition: we have hit EOS
        if (next == 2) { 
            printf(" <EOS>\n");
            break; 
        }
        
        // Print the token as string, decode the token id to string
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        // if (pos % 10 == 0) printf(" [pos:%d]", pos);
        fflush(stdout);
        
        token = next;
        
        // Init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    
    // Report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Achieved tok/s: %.2f\n", (pos-1) / (double)(end-start) * 1000);
    }
    
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// Main generation function implementation

// Fix 6: Updated generate function with debugging
void generate_debug(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    printf("=== DEBUG GENERATION ===\n");
    
    // Encode the prompt
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt ? prompt : "") + 3) * sizeof(int));
    if (prompt) {
        encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    }
    if (num_prompt_tokens < 1) {
        prompt_tokens[0] = 1; // BOS
        num_prompt_tokens = 1;
    }
    
    printf("Prompt tokens: [");
    for (int i = 0; i < num_prompt_tokens; i++) {
        printf("%d", prompt_tokens[i]);
        if (i < num_prompt_tokens - 1) printf(", ");
    }
    printf("]\n");
    
    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    while (pos < steps) {
        printf("\n--- Step %d ---\n", pos);
        printf("Input token: %d (\"%s\")\n", token, tokenizer->vocab[token]);
        
        // Forward pass
        float* logits = forward_fixed(transformer, token, pos);
        
        // Sample next token
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
            printf("Using prompt token: %d\n", next);
        } else {
            next = sample_debug(sampler, logits);
        }
        
        pos++;
        
        if (next == 2) { 
            printf("Hit EOS token, stopping.\n");
            break; 
        }
        
        char* piece = decode(tokenizer, token, next);
        printf("Output: \"%s\"\n", piece);
        safe_printf(piece);
        fflush(stdout);
        
        token = next;
        
        if (start == 0) { start = time_in_ms(); }
    }
    
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Achieved tok/s: %.2f\n", (pos-1) / (double)(end-start) * 1000);
    }
    
    free(prompt_tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}


// ----------------------------------------------------------------------------
// Main function

int main(int argc, char *argv[]) {
    // Initialize logging system
    init_logging("llama2_detailed_log.txt");
    
    // Initialize CUDA with detailed error reporting
    printf("Initializing CUDA...\n");
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    
    printf("CUDA Status: %s\n", cudaGetErrorString(cudaStatus));
    printf("Device Count: %d\n", deviceCount);
    
    bool cuda_available = (cudaStatus == cudaSuccess && deviceCount > 0);
    
    if (!cuda_available) {
        fprintf(stderr, "Warning: No CUDA devices found or CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        fprintf(stderr, "Falling back to CPU mode.\n");
    } else {
        printf("Found %d CUDA device(s)\n", deviceCount);
        CUDA_CHECK(cudaSetDevice(0));
        
        // Print device info
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("Using GPU: %s\n", prop.name);
        printf("GPU Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Max Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("Max Thread Blocks per Grid (per dimension): %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Warp Size: %d\n", prop.warpSize);

        printf("Initializing CUDA libraries...\n");
        // cublasCreate(&cublas_handle);
        // cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH); // Enable Tensor Cores
        printf("CUDA initialization complete!\n");
        
        // Validate optimization requirements
        if (prop.sharedMemPerBlock < 48 * 1024) {
            printf("Warning: Limited shared memory may affect performance\n");
        }
        if (prop.major < 7) {
            printf("Warning: GPU compute capability < 7.0 may have reduced performance\n");
        }
    }
    
    // Set global CUDA availability
    set_cuda_availability(cuda_available);

    // Parse command line arguments
    char *checkpoint_path = NULL;
    const char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    // Parameter validation
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // Build the Transformer
    printf("Loading model...\n");
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, cuda_available);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    // Print model info
    printf("Model loaded successfully!\n");
    printf("Model parameters:\n");
    printf("  Dimensions: %d\n", transformer.config.dim);
    printf("  Layers: %d\n", transformer.config.n_layers);
    printf("  Heads: %d\n", transformer.config.n_heads);
    printf("  KV Heads: %d\n", transformer.config.n_kv_heads);
    printf("  Vocab size: %d\n", transformer.config.vocab_size);
    printf("  Sequence length: %d\n", transformer.config.seq_len);
    printf("  Head size: %d\n", transformer.config.dim / transformer.config.n_heads);
    printf("  Hidden dimension: %d\n", transformer.config.hidden_dim);

    // Build the Tokenizer
    printf("Loading tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // Build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // Print generation parameters
    printf("Generation parameters:\n");
    printf("  Temperature: %.2f\n", temperature);
    printf("  Top-p: %.2f\n", topp);
    printf("  Steps: %d\n", steps);
    printf("  Random seed: %llu\n", rng_seed);
    
#if USE_CUDA
    if (cuda_available) {
        printf("Flash Attention: ULTRA-OPTIMIZED GPU ACCELERATION\n");
    } else {
        printf("Flash Attention: DISABLED (No CUDA device)\n");
    }
#else
    printf("Flash Attention: DISABLED (CPU mode)\n");
#endif
    printf("\n");

    // Run generation with fused transformer optimization
    if (cuda_available) {
        printf("Using FUSED TRANSFORMER kernels for maximum performance!\n");
        generate_fused_optimized(&transformer, &tokenizer, &sampler, prompt, steps);
    } else {
        printf("CUDA not available, using standard CPU approach\n");
        generate_debug(&transformer, &tokenizer, &sampler, prompt, steps);
    }

    // if (cuda_available) {
    //     printf("CUDA available, but using DEBUG mode for troubleshooting!\n");
    //     printf("Using CPU-based debug generation to identify issues...\n");
    //     generate_debug(&transformer, &tokenizer, &sampler, prompt, steps);
        
    //     // Once debugging is complete, you can switch back to:
    //     // generate_fused_optimized(&transformer, &tokenizer, &sampler, prompt, steps);
    // } else {
    //     printf("CUDA not available, using DEBUG CPU approach\n");
    //     generate_debug(&transformer, &tokenizer, &sampler, prompt, steps);
    // }

    // Cleanup
    printf("\nCleaning up...\n");
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    // Cleanup CUDA
    // if (cuda_available) {
    //     cublasDestroy(cublas_handle);
    // }
    
    printf("Done.\n");
    
    // Close logging system
    close_logging();
    
    return 0;
}
