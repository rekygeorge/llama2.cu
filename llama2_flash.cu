/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <float.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

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

// Forward declarations for structs
struct GPUCapabilities;
struct GPUConfig;

// Forward declarations for functions (will be properly declared after struct definitions)
void cuda_rmsnorm_adaptive(float* output, float* input, float* weight, int size);
void cuda_matmul(float* d_xout, float* d_x, float* d_w, int n, int d);
// void cuda_matmul_tiled(float* d_xout, float* d_x, float* d_w, int n, int d);
// void cuda_matmul_coarsened(float* d_xout, float* d_x, float* d_w, int n, int d);
void cuda_matmul_optimized(float* d_xout, float* d_x, float* d_w, int n, int d);
void cuda_swiglu(float* d_hb, float* d_hb2, int size);

GPUCapabilities detect_gpu_capabilities();
GPUConfig detect_gpu_config();
void configure_shared_memory_limits();


#define COARSE_FACTOR 2
#define TILE_SIZE 32  // 64 < 32 --> Better

// Step 1: Tiled Matrix Multiplication with configurable shared memory
#define TILE_SIZE_SMALL 32   // For 48KB config
#define TILE_SIZE_LARGE 64   // For 96KB+ config
// #define TILE_SIZE TILE_SIZE_LARGE

// // Global variables for CUDA
// static cublasHandle_t cublas_handle;
static bool g_cuda_available = false;
static FILE* g_log_file = NULL;
static bool g_enable_profile = false;
static bool g_profile_triggered = false;
static int g_profile_pos = 10;
static char g_profile_csv_path[512] = "flash_profile_metrics.csv";

// Global cuBLAS handle (initialize once at program start)
cublasHandle_t cublas_handle;

void init_cublas() {
    cublasCreate(&cublas_handle);
    // Optional: set math mode for better performance on newer GPUs
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
}

void cleanup_cublas() {
    cublasDestroy(cublas_handle);
}

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

void append_profile_metrics_csv(
    int token,
    int pos,
    int dim,
    int hidden_dim,
    int n_heads,
    int n_kv_heads,
    float stage1_ms,
    float stage2_ms,
    float stage3_ms,
    float stage4_ms,
    float stage5_ms,
    float stage6_ms,
    float stage7_ms,
    float stage8_ms,
    float total_layer_ms,
    float est_32_layer_ms,
    float est_tok_s
) {
    int needs_header = 0;
    FILE* check_file = fopen(g_profile_csv_path, "r");
    if (!check_file) {
        needs_header = 1;
    } else {
        fclose(check_file);
    }

    FILE* fp = fopen(g_profile_csv_path, "a");
    if (!fp) {
        fprintf(stderr, "Warning: Could not open profile CSV '%s' for append\n", g_profile_csv_path);
        return;
    }

    if (needs_header) {
        fprintf(fp,
                "timestamp_unix,token,pos,dim,hidden_dim,n_heads,n_kv_heads,"
                "rms_att_ms,qkv_ms,rope_kvcache_ms,flash_attn_ms,out_proj_res_ms,"
                "rms_ffn_ms,ffn_w1w3_ms,swiglu_w2_ms,total_layer_ms,est_32_layer_ms,est_tok_s\n");
    }

    fprintf(fp,
            "%lld,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            (long long)time(NULL),
            token,
            pos,
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            stage1_ms,
            stage2_ms,
            stage3_ms,
            stage4_ms,
            stage5_ms,
            stage6_ms,
            stage7_ms,
            stage8_ms,
            total_layer_ms,
            est_32_layer_ms,
            est_tok_s);
    fclose(fp);
}


// ----------------------------------------------------------------------------
// GPU capability detection
struct GPUCapabilities {
    int major;
    int minor;
    int multiProcessorCount;
    size_t sharedMemPerBlock;
    int maxThreadsPerBlock;
    char name[256];
};

// GPU detection structure
struct GPUConfig {
    int major, minor;
    int sm_count;
    size_t shared_mem_per_block;
    int max_threads_per_block;
    bool is_a100;
    bool is_rtx20_series;
};


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

// Single matrix multiplication using cuBLAS
void cublas_matmul(float* d_out, float* d_x, float* d_w, int n, int d) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS uses column-major format, so we need to transpose our operation
    // We want: out = W * x (where W is d×n, x is n×1, out is d×1)
    // In cuBLAS terms: out = W * x
    cublasSgemv(cublas_handle, CUBLAS_OP_N,
                d, n,           // dimensions: d rows, n columns
                &alpha,         // scaling factor
                d_w, d,         // matrix W with leading dimension d
                d_x, 1,         // vector x with stride 1
                &beta,          // scaling factor for output
                d_out, 1);      // output vector with stride 1
}

// Element-wise kernels (these were working fine)
__global__ void elementwise_add_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

// Simple, reliable matrix multiplication kernel
__global__ void matmul_kernel_simple(float* xout, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < d) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Step 1: Tiled Matrix Multiplication
// This version uses shared memory to cache frequently accessed data

__global__ void matmul_kernel_tiled(float* xout, float* x, float* w, int n, int d) {
    // Shared memory for caching input vector tiles
    __shared__ float shared_x[TILE_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    // Process input vector in tiles
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // Load tile of input vector into shared memory
        if (tid < TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE) {
            shared_x[tid] = 0.0f;  // Padding
        }
        
        __syncthreads();
        
        // Compute partial dot product using shared memory
        int tile_end = min(TILE_SIZE, n - tile_start);
        for (int j = 0; j < tile_end; j++) {
            val += w[i * n + tile_start + j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

// Step 2: Thread Coarsening - Each thread computes multiple output elements


__global__ void matmul_kernel_coarsened(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[TILE_SIZE];
    
    int tid = threadIdx.x;
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    // Each thread computes COARSE_FACTOR output elements
    float vals[COARSE_FACTOR] = {0.0f};
    
    // Process input vector in tiles
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // Load tile of input vector into shared memory
        if (tid < TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products for multiple outputs
        int tile_end = min(TILE_SIZE, n - tile_start);
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int i = base_i + c;
            if (i < d) {
                for (int j = 0; j < tile_end; j++) {
                    vals[c] += w[i * n + tile_start + j] * shared_x[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = base_i + c;
        if (i < d) {
            xout[i] = vals[c];
        }
    }
}

// Step 3: Advanced optimization with vectorized loads and better memory access
__global__ void matmul_kernel_optimized(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[TILE_SIZE];
    
    int tid = threadIdx.x;
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    float vals[COARSE_FACTOR] = {0.0f};
    
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        // Coalesced load into shared memory
        if (tid < TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(TILE_SIZE, n - tile_start);
        
        // Unrolled inner loop for better performance
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int i = base_i + c;
            if (i < d) {
                const float* w_row = &w[i * n + tile_start];
                #pragma unroll 8
                for (int j = 0; j < tile_end; j++) {
                    vals[c] += w_row[j] * shared_x[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results with bounds checking
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = base_i + c;
        if (i < d) {
            xout[i] = vals[c];
        }
    }
}

// Additional optimization for specific dimensions - 2D thread blocks
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define IMPROVED_TILE_SIZE 32

// Version 1: Enhanced 2D with larger tiles and coarsening
__global__ void matmul_kernel_2d_improved(float* xout, float* x, float* w, int n, int d) {
    // Larger shared memory for better reuse
    __shared__ float shared_x[IMPROVED_TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;  // Linear thread ID
    
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    for (int tile_start = 0; tile_start < n; tile_start += IMPROVED_TILE_SIZE) {
        // Collaborative loading with all 256 threads
        if (tid < IMPROVED_TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < IMPROVED_TILE_SIZE) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(IMPROVED_TILE_SIZE, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        // Optimized computation with unrolling
        #pragma unroll 8
        for (int j = 0; j < tile_end; j++) {
            val += w_row[j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

// Version 2: 2D with vectorized loads (float4)
__global__ void matmul_kernel_2d_vectorized(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[IMPROVED_TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    for (int tile_start = 0; tile_start < n; tile_start += IMPROVED_TILE_SIZE) {
        // Vectorized loading where possible
        if (tid < IMPROVED_TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < IMPROVED_TILE_SIZE) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(IMPROVED_TILE_SIZE, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        // Vectorized computation when tile_end is multiple of 4
        int vec_end = (tile_end / 4) * 4;
        for (int j = 0; j < vec_end; j += 4) {
            // Manual vectorization
            val += w_row[j] * shared_x[j] + 
                   w_row[j+1] * shared_x[j+1] + 
                   w_row[j+2] * shared_x[j+2] + 
                   w_row[j+3] * shared_x[j+3];
        }
        
        // Handle remaining elements
        for (int j = vec_end; j < tile_end; j++) {
            val += w_row[j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

// Version 3: 2D with optimal tile size for your dimensions
__global__ void matmul_kernel_2d_optimized_4096(float* xout, float* x, float* w, int n, int d) {
    // Optimized for 4096: 4096/64 = 64 clean iterations
    __shared__ float shared_x[64];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    // 64 iterations for 4096 dimension - perfect fit
    for (int tile_start = 0; tile_start < n; tile_start += 64) {
        if (tid < 64 && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < 64) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(64, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        // Fully unrolled for 64 elements (or use partial unroll)
        #pragma unroll 16
        for (int j = 0; j < tile_end; j++) {
            val += w_row[j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

// Version 4: 2D with warp-level optimizations
__global__ void matmul_kernel_2d_warp_optimized(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[64];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    int warpId = tid / 32;
    int laneId = tid % 32;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    for (int tile_start = 0; tile_start < n; tile_start += 64) {
        // Warp-coalesced loading
        if (tid < 64 && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < 64) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(64, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        // Process in warp-sized chunks for better instruction throughput
        for (int j = 0; j < tile_end; j += 32) {
            int chunk_end = min(32, tile_end - j);
            #pragma unroll 8
            for (int k = 0; k < chunk_end; k++) {
                val += w_row[j + k] * shared_x[j + k];
            }
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

// Host functions for improved 2D versions
void cuda_matmul_2d_improved(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y / 2);  // /2 for coarsening
    
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_2d_improved, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_2d_improved<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul_2d_vectorized(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_2d_vectorized, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_2d_vectorized<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul_2d_optimized_4096(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_2d_optimized_4096, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_2d_optimized_4096<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul_2d_warp_optimized(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_2d_warp_optimized, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_2d_warp_optimized<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Ultimate 2D version - combines best techniques
__global__ void matmul_kernel_2d_ultimate(float* xout, float* x, float* w, int n, int d) {
    // Optimal tile size for 4096 and efficient shared memory usage
    __shared__ float shared_x[128];  // 512 bytes - efficient size
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    // Process in 128-element tiles (4096/128 = 32 iterations)
    for (int tile_start = 0; tile_start < n; tile_start += 128) {
        // All 256 threads participate in loading
        if (tid < 128 && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < 128) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(128, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        // Optimized inner loop with multiple accumulation
        float acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f;
        
        int vec_end = (tile_end / 4) * 4;
        for (int j = 0; j < vec_end; j += 4) {
            acc1 += w_row[j] * shared_x[j];
            acc2 += w_row[j+1] * shared_x[j+1];
            acc3 += w_row[j+2] * shared_x[j+2];
            acc4 += w_row[j+3] * shared_x[j+3];
        }
        
        val += acc1 + acc2 + acc3 + acc4;
        
        // Handle remaining elements
        for (int j = vec_end; j < tile_end; j++) {
            val += w_row[j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}

void cuda_matmul_2d_ultimate(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    // Configure for maximum shared memory performance
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_2d_ultimate, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_2d_ultimate<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void matmul_kernel_2d_tiled(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_SIZE_X + tx;
    
    int i = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    for (int tile_start = 0; tile_start < n; tile_start += BLOCK_SIZE_X * BLOCK_SIZE_Y) {
        // Load tile collaboratively
        if (tid < BLOCK_SIZE_X * BLOCK_SIZE_Y && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < BLOCK_SIZE_X * BLOCK_SIZE_Y) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(BLOCK_SIZE_X * BLOCK_SIZE_Y, n - tile_start);
        for (int j = 0; j < tile_end; j++) {
            val += w[i * n + tile_start + j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}


void cuda_matmul_2d_tiled(float* d_xout, float* d_x, float* d_w, int n, int d) {
    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(1, (d + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    matmul_kernel_2d_tiled<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Enhanced version with larger shared memory allocation
__global__ void matmul_kernel_large_shared(float* xout, float* x, float* w, int n, int d) {
    // Use larger shared memory for input vector only (safer approach)
    __shared__ float shared_x[TILE_SIZE_LARGE];
    
    int tid = threadIdx.x;
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    float vals[COARSE_FACTOR] = {0.0f};
    
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE_LARGE) {
        // Load input vector tile cooperatively
        if (tid < TILE_SIZE_LARGE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE_LARGE) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(TILE_SIZE_LARGE, n - tile_start);
        
        // Compute partial dot products for multiple outputs
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int i = base_i + c;
            if (i < d) {
                const float* w_row = &w[i * n + tile_start];
                #pragma unroll 8
                for (int j = 0; j < tile_end; j++) {
                    vals[c] += w_row[j] * shared_x[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = base_i + c;
        if (i < d) {
            xout[i] = vals[c];
        }
    }
}

// Host functions with shared memory configuration
void cuda_matmul_with_shared_config(float* d_xout, float* d_x, float* d_w, int n, int d) {
    // Use the safer fixed version
    int blockSize = 256;
    int numBlocks = ((d + COARSE_FACTOR - 1) / COARSE_FACTOR + blockSize - 1) / blockSize;
    
    // Configure for larger shared memory but don't exceed safe limits
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_large_shared, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_large_shared<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Alternative: Simple but fast version with proper tile sizing
__global__ void matmul_kernel_simple_fast(float* xout, float* x, float* w, int n, int d) {
    // Use tile size that divides evenly into common dimensions
    __shared__ float shared_x[128];  // Conservative size for broad compatibility
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= d) return;
    
    float val = 0.0f;
    
    for (int tile_start = 0; tile_start < n; tile_start += 128) {
        // Load tile collaboratively
        if (tid < 128 && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < 128) {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        int tile_end = min(128, n - tile_start);
        const float* w_row = &w[i * n + tile_start];
        
        #pragma unroll 8
        for (int j = 0; j < tile_end; j++) {
            val += w_row[j] * shared_x[j];
        }
        
        __syncthreads();
    }
    
    xout[i] = val;
}


void cuda_matmul_simple_fast(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 256;
    int numBlocks = (d + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_simple_fast, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_simple_fast<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}



// Fixed high-performance version - optimized for your use case
__global__ void matmul_kernel_optimized_fixed(float* xout, float* x, float* w, int n, int d) {
    // Optimize tile size for block size (256 threads)
    __shared__ float shared_x[256];  // Match block size for full utilization
    
    int tid = threadIdx.x;
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    float vals[COARSE_FACTOR] = {0.0f};
    
    // Process input in tiles that match our thread count
    for (int tile_start = 0; tile_start < n; tile_start += 256) {
        // Every thread loads one element - full coalesced access
        if (tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else {
            shared_x[tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(256, n - tile_start);
        
        // Compute partial dot products - unroll aggressively
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int i = base_i + c;
            if (i < d) {
                const float* w_row = &w[i * n + tile_start];
                float partial = 0.0f;
                
                // Unroll inner loop completely for small tile_end
                #pragma unroll 16
                for (int j = 0; j < tile_end; j++) {
                    partial += w_row[j] * shared_x[j];
                }
                vals[c] += partial;
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = base_i + c;
        if (i < d) {
            xout[i] = vals[c];
        }
    }
}


void cuda_matmul_optimized_fixed(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 256;  // Optimal for most GPUs
    int numBlocks = ((d + COARSE_FACTOR - 1) / COARSE_FACTOR + blockSize - 1) / blockSize;
    
    // Set cache preference for shared memory
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_optimized_fixed, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_optimized_fixed<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Alternative: Safer large shared memory version with double buffering
__global__ void matmul_kernel_double_buffered(float* xout, float* x, float* w, int n, int d) {
    // Double buffering for input vector - safer than weight caching
    __shared__ float shared_x[2][TILE_SIZE_LARGE];
    
    int tid = threadIdx.x;
    int base_i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    
    float vals[COARSE_FACTOR] = {0.0f};
    
    int current_buffer = 0;
    
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE_LARGE) {
        // Load into current buffer
        if (tid < TILE_SIZE_LARGE && tile_start + tid < n) {
            shared_x[current_buffer][tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE_LARGE) {
            shared_x[current_buffer][tid] = 0.0f;
        }
        
        __syncthreads();
        
        int tile_end = min(TILE_SIZE_LARGE, n - tile_start);
        
        // Compute using current buffer
        #pragma unroll
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int i = base_i + c;
            if (i < d) {
                const float* w_row = &w[i * n + tile_start];
                #pragma unroll 8
                for (int j = 0; j < tile_end; j++) {
                    vals[c] += w_row[j] * shared_x[current_buffer][j];
                }
            }
        }
        
        __syncthreads();
        
        // Switch buffer for next iteration
        current_buffer = 1 - current_buffer;
    }
    
    // Write results
    #pragma unroll
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int i = base_i + c;
        if (i < d) {
            xout[i] = vals[c];
        }
    }
}

// Alternative safer version with double buffering
void cuda_matmul_double_buffered(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 256;
    int numBlocks = ((d + COARSE_FACTOR - 1) / COARSE_FACTOR + blockSize - 1) / blockSize;
    
    // Configure shared memory preference
    CUDA_CHECK(cudaFuncSetCacheConfig(matmul_kernel_double_buffered, 
                                     cudaFuncCachePreferShared));
    
    matmul_kernel_double_buffered<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------

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
__global__ void rmsnorm_kernel(float* output, float* input, float* weight, int size, float eps) {
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
// __global__ void swiglu_kernel(float* output, float* gate, float* up, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         float x = gate[idx];
//         float silu = x / (1.0f + expf(-x)); // SiLU activation
//         output[idx] = silu * up[idx];
//     }
// }

__global__ void swiglu_kernel(float* hb, float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}

// SwiGLU activation CUDA kernel
__global__ void swiglu_kernel_cublas_v2(float* output, float* gate, float* up, int size) {
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

// ----------------------------------------------------------------------------

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
    
    CUDA_CHECK(cudaDeviceSynchronize());

#endif
}


// ----------------------------------------------------------------------------// Fused QKV projection kernel
// Fix 1: Corrected fused QKV projection kernel
// Fused QKV projection kernel - computes Q, K, V in a single kernel launch
__global__ void fused_qkv_kernel(
    const float* input,     // Input vector [dim]
    const float* Wq,        // Query weight matrix [dim x dim] 
    const float* Wk,        // Key weight matrix [dim x kv_dim]
    const float* Wv,        // Value weight matrix [dim x kv_dim]
    float* q_out,           // Query output [dim]
    float* k_out,           // Key output [kv_dim] 
    float* v_out,           // Value output [kv_dim]
    const int dim,          // Input/Query dimension
    const int kv_dim,       // Key/Value dimension
    const float alpha,      // Scaling factor
    const float beta        // Bias factor (typically 0)
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input vector (reused across Q, K, V computations)
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    // Load input vector into shared memory cooperatively
    for (int i = tid; i < dim; i += blockDim.x) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Block 0: Compute Q = input * Wq^T
    if (bid == 0) {
        // Each thread computes one output element of Q
        for (int out_idx = tid; out_idx < dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Compute dot product: Q[out_idx] = sum(input[i] * Wq[out_idx, i])
            // Wq is stored in column-major: Wq[out_idx, i] = Wq[out_idx * dim + i]
            for (int i = 0; i < dim; i++) {
                sum += input_shared[i] * Wq[out_idx * dim + i];
            }
            
            q_out[out_idx] = alpha * sum + beta * q_out[out_idx];
        }
    }
    
    // Block 1: Compute K = input * Wk^T  
    else if (bid == 1) {
        // Each thread computes one output element of K
        for (int out_idx = tid; out_idx < kv_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Compute dot product: K[out_idx] = sum(input[i] * Wk[out_idx, i])
            // Wk is stored in column-major: Wk[out_idx, i] = Wk[out_idx * dim + i]
            for (int i = 0; i < dim; i++) {
                sum += input_shared[i] * Wk[out_idx * dim + i];
            }
            
            k_out[out_idx] = alpha * sum + beta * k_out[out_idx];
        }
    }
    
    // Block 2: Compute V = input * Wv^T
    else if (bid == 2) {
        // Each thread computes one output element of V
        for (int out_idx = tid; out_idx < kv_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Compute dot product: V[out_idx] = sum(input[i] * Wv[out_idx, i])
            // Wv is stored in column-major: Wv[out_idx, i] = Wv[out_idx * dim + i]
            for (int i = 0; i < dim; i++) {
                sum += input_shared[i] * Wv[out_idx * dim + i];
            }
            
            v_out[out_idx] = alpha * sum + beta * v_out[out_idx];
        }
    }
}

// A100-optimized QKV kernel with maximum parallelization
// v2 - Fused QKV projection kernel - computes Q, K, V in a single kernel launch

__global__ void qkv_kernel_a100_optimized(
    const float* __restrict__ input,     // [dim]
    const float* __restrict__ Wq,        // [dim x dim] 
    const float* __restrict__ Wk,        // [dim x kv_dim]
    const float* __restrict__ Wv,        // [dim x kv_dim]
    float* __restrict__ q_out,           // [dim]
    float* __restrict__ k_out,           // [kv_dim]
    float* __restrict__ v_out,           // [kv_dim]
    const int dim,
    const int kv_dim,
    const float alpha,
    const float beta
) {
    // A100: Use large blocks and vectorized operations
    extern __shared__ float smem[];
    
    // Partition shared memory efficiently for A100's 164KB (now using up to 140KB)
    float* input_shared = smem;                    // [dim]
    float* weight_cache = input_shared + dim;      // [LARGE_CACHE_SIZE] for aggressive weight prefetching
    const int cache_size = 8192;  // Increased cache size for A100
    float* temp_results = weight_cache + cache_size; // [additional workspace]
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Cooperatively load input vector (vectorized for A100)
    const int vec_size = 4;  // 128-bit loads
    const int vectorized_dim = (dim + vec_size - 1) / vec_size;
    
    for (int i = tid; i < vectorized_dim; i += block_size) {
        int base_idx = i * vec_size;
        if (base_idx + 3 < dim) {
            // Vectorized load (4 floats at once)
            float4 input_vec = *reinterpret_cast<const float4*>(&input[base_idx]);
            *reinterpret_cast<float4*>(&input_shared[base_idx]) = input_vec;
        } else {
            // Handle remaining elements
            for (int j = 0; j < vec_size && base_idx + j < dim; j++) {
                input_shared[base_idx + j] = input[base_idx + j];
            }
        }
    }
    __syncthreads();
    
    // Process different projections in parallel blocks
    if (bid == 0) {
        // Block 0: Compute Q projection with A100 optimizations
        const int outputs_per_thread = (dim + block_size - 1) / block_size;
        
        for (int t = 0; t < outputs_per_thread; t++) {
            int out_idx = tid + t * block_size;
            if (out_idx < dim) {
                float sum = 0.0f;
                
                // Unrolled inner loop for better ILP (Instruction Level Parallelism)
                int i = 0;
                for (; i + 15 < dim; i += 16) {  // Process 16 elements at once
                    // Manually unroll for better performance on A100
                    sum += input_shared[i+0] * Wq[out_idx * dim + i+0];
                    sum += input_shared[i+1] * Wq[out_idx * dim + i+1];
                    sum += input_shared[i+2] * Wq[out_idx * dim + i+2];
                    sum += input_shared[i+3] * Wq[out_idx * dim + i+3];
                    sum += input_shared[i+4] * Wq[out_idx * dim + i+4];
                    sum += input_shared[i+5] * Wq[out_idx * dim + i+5];
                    sum += input_shared[i+6] * Wq[out_idx * dim + i+6];
                    sum += input_shared[i+7] * Wq[out_idx * dim + i+7];
                    sum += input_shared[i+8] * Wq[out_idx * dim + i+8];
                    sum += input_shared[i+9] * Wq[out_idx * dim + i+9];
                    sum += input_shared[i+10] * Wq[out_idx * dim + i+10];
                    sum += input_shared[i+11] * Wq[out_idx * dim + i+11];
                    sum += input_shared[i+12] * Wq[out_idx * dim + i+12];
                    sum += input_shared[i+13] * Wq[out_idx * dim + i+13];
                    sum += input_shared[i+14] * Wq[out_idx * dim + i+14];
                    sum += input_shared[i+15] * Wq[out_idx * dim + i+15];
                }
                
                // Handle remaining elements
                for (; i < dim; i++) {
                    sum += input_shared[i] * Wq[out_idx * dim + i];
                }
                
                q_out[out_idx] = alpha * sum + beta * q_out[out_idx];
            }
        }
    }
    else if (bid == 1) {
        // Block 1: Compute K projection
        const int outputs_per_thread = (kv_dim + block_size - 1) / block_size;
        
        for (int t = 0; t < outputs_per_thread; t++) {
            int out_idx = tid + t * block_size;
            if (out_idx < kv_dim) {
                float sum = 0.0f;
                
                // Same unrolled optimization for K
                int i = 0;
                for (; i + 15 < dim; i += 16) {
                    sum += input_shared[i+0] * Wk[out_idx * dim + i+0];
                    sum += input_shared[i+1] * Wk[out_idx * dim + i+1];
                    sum += input_shared[i+2] * Wk[out_idx * dim + i+2];
                    sum += input_shared[i+3] * Wk[out_idx * dim + i+3];
                    sum += input_shared[i+4] * Wk[out_idx * dim + i+4];
                    sum += input_shared[i+5] * Wk[out_idx * dim + i+5];
                    sum += input_shared[i+6] * Wk[out_idx * dim + i+6];
                    sum += input_shared[i+7] * Wk[out_idx * dim + i+7];
                    sum += input_shared[i+8] * Wk[out_idx * dim + i+8];
                    sum += input_shared[i+9] * Wk[out_idx * dim + i+9];
                    sum += input_shared[i+10] * Wk[out_idx * dim + i+10];
                    sum += input_shared[i+11] * Wk[out_idx * dim + i+11];
                    sum += input_shared[i+12] * Wk[out_idx * dim + i+12];
                    sum += input_shared[i+13] * Wk[out_idx * dim + i+13];
                    sum += input_shared[i+14] * Wk[out_idx * dim + i+14];
                    sum += input_shared[i+15] * Wk[out_idx * dim + i+15];
                }
                
                for (; i < dim; i++) {
                    sum += input_shared[i] * Wk[out_idx * dim + i];
                }
                
                k_out[out_idx] = alpha * sum + beta * k_out[out_idx];
            }
        }
    }
    else if (bid == 2) {
        // Block 2: Compute V projection
        const int outputs_per_thread = (kv_dim + block_size - 1) / block_size;
        
        for (int t = 0; t < outputs_per_thread; t++) {
            int out_idx = tid + t * block_size;
            if (out_idx < kv_dim) {
                float sum = 0.0f;
                
                // Same unrolled optimization for V
                int i = 0;
                for (; i + 15 < dim; i += 16) {
                    sum += input_shared[i+0] * Wv[out_idx * dim + i+0];
                    sum += input_shared[i+1] * Wv[out_idx * dim + i+1];
                    sum += input_shared[i+2] * Wv[out_idx * dim + i+2];
                    sum += input_shared[i+3] * Wv[out_idx * dim + i+3];
                    sum += input_shared[i+4] * Wv[out_idx * dim + i+4];
                    sum += input_shared[i+5] * Wv[out_idx * dim + i+5];
                    sum += input_shared[i+6] * Wv[out_idx * dim + i+6];
                    sum += input_shared[i+7] * Wv[out_idx * dim + i+7];
                    sum += input_shared[i+8] * Wv[out_idx * dim + i+8];
                    sum += input_shared[i+9] * Wv[out_idx * dim + i+9];
                    sum += input_shared[i+10] * Wv[out_idx * dim + i+10];
                    sum += input_shared[i+11] * Wv[out_idx * dim + i+11];
                    sum += input_shared[i+12] * Wv[out_idx * dim + i+12];
                    sum += input_shared[i+13] * Wv[out_idx * dim + i+13];
                    sum += input_shared[i+14] * Wv[out_idx * dim + i+14];
                    sum += input_shared[i+15] * Wv[out_idx * dim + i+15];
                }
                
                for (; i < dim; i++) {
                    sum += input_shared[i] * Wv[out_idx * dim + i];
                }
                
                v_out[out_idx] = alpha * sum + beta * v_out[out_idx];
            }
        }
    }
}

// RTX 2070 Super optimized kernel (conservative approach)
__global__ void qkv_kernel_rtx2070_optimized(
    const float* __restrict__ input,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const int dim,
    const int kv_dim,
    const float alpha,
    const float beta
) {
    // RTX 2070 Super: Enhanced approach with 96KB shared memory limit
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    // Load input vector (2-element vectorization for RTX 2070 Super)
    for (int i = tid * 2; i < dim; i += block_size * 2) {
        if (i < dim) input_shared[i] = input[i];
        if (i + 1 < dim) input_shared[i + 1] = input[i + 1];
    }
    __syncthreads();
    
    // Same structure as A100 but with less aggressive unrolling
    if (bid == 0) {
        // Q projection with 4-way unrolling (conservative for RTX 2070 Super)
        for (int out_idx = tid; out_idx < dim; out_idx += block_size) {
            float sum = 0.0f;
            
            int i = 0;
            for (; i + 3 < dim; i += 4) {  // 4-way unrolling instead of 16-way
                sum += input_shared[i+0] * Wq[out_idx * dim + i+0];
                sum += input_shared[i+1] * Wq[out_idx * dim + i+1];
                sum += input_shared[i+2] * Wq[out_idx * dim + i+2];
                sum += input_shared[i+3] * Wq[out_idx * dim + i+3];
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wq[out_idx * dim + i];
            }
            
            q_out[out_idx] = alpha * sum + beta * q_out[out_idx];
        }
    }
    else if (bid == 1) {
        // K projection
        for (int out_idx = tid; out_idx < kv_dim; out_idx += block_size) {
            float sum = 0.0f;
            
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                sum += input_shared[i+0] * Wk[out_idx * dim + i+0];
                sum += input_shared[i+1] * Wk[out_idx * dim + i+1];
                sum += input_shared[i+2] * Wk[out_idx * dim + i+2];
                sum += input_shared[i+3] * Wk[out_idx * dim + i+3];
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wk[out_idx * dim + i];
            }
            
            k_out[out_idx] = alpha * sum + beta * k_out[out_idx];
        }
    }
    else if (bid == 2) {
        // V projection
        for (int out_idx = tid; out_idx < kv_dim; out_idx += block_size) {
            float sum = 0.0f;
            
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                sum += input_shared[i+0] * Wv[out_idx * dim + i+0];
                sum += input_shared[i+1] * Wv[out_idx * dim + i+1];
                sum += input_shared[i+2] * Wv[out_idx * dim + i+2];
                sum += input_shared[i+3] * Wv[out_idx * dim + i+3];
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wv[out_idx * dim + i];
            }
            
            v_out[out_idx] = alpha * sum + beta * v_out[out_idx];
        }
    }
}

// Multi-block version for maximum A100 utilization
__global__ void qkv_kernel_multiblock_a100(
    const float* __restrict__ input,
    const float* __restrict__ Wq,
    const float* __restrict__ Wk,
    const float* __restrict__ Wv,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const int dim,
    const int kv_dim,
    const float alpha,
    const float beta
) {
    // Use many blocks to saturate A100's 108 SMs
    extern __shared__ float smem[];
    
    // Enhanced shared memory layout for A100 (up to 140KB available)
    float* input_shared = smem;                    // [dim]
    float* weight_tile = input_shared + dim;       // [8192] large weight tile cache
    float* output_buffer = weight_tile + 8192;     // [additional workspace]
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_blocks = gridDim.x;
    const int block_size = blockDim.x;
    
    // Load input cooperatively across all blocks
    for (int i = tid; i < dim; i += block_size) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Distribute work across all blocks
    int projection_type = bid % 3;  // 0=Q, 1=K, 2=V
    int block_in_type = bid / 3;
    
    if (projection_type == 0) {
        // Q projection distributed across multiple blocks
        int outputs_per_block = (dim + (total_blocks/3) - 1) / (total_blocks/3);
        int output_start = block_in_type * outputs_per_block;
        int output_end = min(output_start + outputs_per_block, dim);
        
        for (int out_idx = output_start + tid; out_idx < output_end; out_idx += block_size) {
            float sum = 0.0f;
            
            // Aggressive unrolling for A100
            int i = 0;
            for (; i + 31 < dim; i += 32) {  // Even more aggressive unrolling
                #pragma unroll 32
                for (int j = 0; j < 32; j++) {
                    sum += input_shared[i + j] * Wq[out_idx * dim + i + j];
                }
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wq[out_idx * dim + i];
            }
            
            q_out[out_idx] = alpha * sum + beta * q_out[out_idx];
        }
    }
    else if (projection_type == 1) {
        // K projection (similar structure)
        int outputs_per_block = (kv_dim + (total_blocks/3) - 1) / (total_blocks/3);
        int output_start = block_in_type * outputs_per_block;
        int output_end = min(output_start + outputs_per_block, kv_dim);
        
        for (int out_idx = output_start + tid; out_idx < output_end; out_idx += block_size) {
            float sum = 0.0f;
            
            int i = 0;
            for (; i + 31 < dim; i += 32) {
                #pragma unroll 32
                for (int j = 0; j < 32; j++) {
                    sum += input_shared[i + j] * Wk[out_idx * dim + i + j];
                }
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wk[out_idx * dim + i];
            }
            
            k_out[out_idx] = alpha * sum + beta * k_out[out_idx];
        }
    }
    else {
        // V projection (similar structure)
        int outputs_per_block = (kv_dim + (total_blocks/3) - 1) / (total_blocks/3);
        int output_start = block_in_type * outputs_per_block;
        int output_end = min(output_start + outputs_per_block, kv_dim);
        
        for (int out_idx = output_start + tid; out_idx < output_end; out_idx += block_size) {
            float sum = 0.0f;
            
            int i = 0;
            for (; i + 31 < dim; i += 32) {
                #pragma unroll 32
                for (int j = 0; j < 32; j++) {
                    sum += input_shared[i + j] * Wv[out_idx * dim + i + j];
                }
            }
            
            for (; i < dim; i++) {
                sum += input_shared[i] * Wv[out_idx * dim + i];
            }
            
            v_out[out_idx] = alpha * sum + beta * v_out[out_idx];
        }
    }
}

// Adaptive wrapper function
void cuda_qkv_adaptive_optimized(
    const float* input,
    const float* Wq, const float* Wk, const float* Wv,
    float* q_out, float* k_out, float* v_out,
    const int dim, const int kv_dim,
    const float alpha, const float beta
) {
    GPUConfig config = detect_gpu_config();
    
    size_t smem_size;
    
    if (config.is_a100) {
        // A100: Use much larger shared memory (up to 140KB out of 164KB available)
        smem_size = min((size_t)(140 * 1024), (dim + 8192) * sizeof(float));
    } else {
        // Other GPUs: Conservative approach
        smem_size = (dim + 1024) * sizeof(float);
    }
    
    // Configure shared memory based on GPU type
    if (config.is_a100) {
        // A100: Enable maximum shared memory per block (163 KB)
        cudaFuncSetAttribute(qkv_kernel_a100_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, 163 * 1024);
        cudaFuncSetAttribute(qkv_kernel_multiblock_a100, cudaFuncAttributeMaxDynamicSharedMemorySize, 163 * 1024);
    } else if (config.major >= 8) {
        // RTX 30xx/40xx: Enable large shared memory (99 KB)
        cudaFuncSetAttribute(qkv_kernel_a100_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        cudaFuncSetAttribute(qkv_kernel_multiblock_a100, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
    } else if (config.major == 7) {
        // RTX 20xx/V100: Enable larger shared memory (64-96 KB)
        size_t max_smem = (config.minor == 0) ? 96 * 1024 : 64 * 1024;  // Volta: 96KB, Turing: 64KB
        cudaFuncSetAttribute(qkv_kernel_a100_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
        cudaFuncSetAttribute(qkv_kernel_multiblock_a100, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
    }
    
    if (config.is_a100) {
        if (dim >= 4096) {
            // Large model: Use multi-block approach to saturate all 108 SMs
            int blocks_per_projection = min(config.sm_count / 3, (dim + 1023) / 1024);
            int total_blocks = blocks_per_projection * 3;
            
            dim3 gridDim(total_blocks);
            dim3 blockDim(1024);  // Maximum threads per block
            
            // printf("A100 Multi-block: %d blocks, 1024 threads/block\n", total_blocks);
            
            qkv_kernel_multiblock_a100<<<gridDim, blockDim, smem_size>>>(
                input, Wq, Wk, Wv, q_out, k_out, v_out, dim, kv_dim, alpha, beta
            );
        } else {
            // Smaller model: Use 3-block approach
            dim3 gridDim(3);
            dim3 blockDim(1024);
            
            // printf("A100 QKV: Using %zu KB shared memory for fused kernel\n", smem_size / 1024);
            
            qkv_kernel_a100_optimized<<<gridDim, blockDim, smem_size>>>(
                input, Wq, Wk, Wv, q_out, k_out, v_out, dim, kv_dim, alpha, beta
            );
        }
    }
    else if (config.is_rtx20_series) {
        // RTX 2070 Super: Conservative approach
        dim3 gridDim(3);
        dim3 blockDim(256);  // Conservative for better occupancy
        smem_size = min(smem_size, (size_t)(40 * 1024));  // Stay under 48KB limit
        
        // printf("RTX 2070 Super: 3 blocks, 256 threads/block, %zuKB shared mem\n", smem_size/1024);
        
        qkv_kernel_rtx2070_optimized<<<gridDim, blockDim, smem_size>>>(
            input, Wq, Wk, Wv, q_out, k_out, v_out, dim, kv_dim, alpha, beta
        );
    }
    else {
        // Other GPUs: Fallback to original approach
        dim3 gridDim(3);
        dim3 blockDim(256);
        smem_size = min(smem_size, config.shared_mem_per_block / 2);
        
        qkv_kernel_rtx2070_optimized<<<gridDim, blockDim, smem_size>>>(
            input, Wq, Wk, Wv, q_out, k_out, v_out, dim, kv_dim, alpha, beta
        );
    }
}




// ----------------------------------------------------------------------------// CUDA kernel for single query Flash Attention
// This kernel processes a single query against the key and

__global__ void single_query_flashattention_kernel(
    const float* q,         // Single query [head_size]
    const float* k_cache,   // Key cache [seq_len, head_size] with stride
    const float* v_cache,   // Value cache [seq_len, head_size] with stride
    float* output,          // Output [head_size]
    const int seq_len,      // Current sequence length (pos + 1)
    const int head_size,    // Head dimension
    const int Bc,           // Column block size
    const int kv_stride,    // Stride between positions in K,V cache
    const float scale       // Attention scale (1/sqrt(head_size))
) {
    extern __shared__ float smem[];
    
    // Shared memory layout
    float* Kj = smem;                           // [Bc, head_size]
    float* Vj = Kj + Bc * head_size;           // [Bc, head_size]
    float* Qi = Vj + Bc * head_size;           // [head_size]
    float* Sij = Qi + head_size;               // [Bc]
    float* Pij = Sij + Bc;                     // [Bc]
    float* Oi_local = Pij + Bc;                // [head_size]
    float* stats = Oi_local + head_size;       // [6] for mi, li, etc.
    
    const int tid = threadIdx.x;
    
    // Initialize statistics
    if (tid == 0) {
        stats[0] = -FLT_MAX;  // mi (max)
        stats[1] = 0.0f;      // li (sum)
    }
    
    // Load query into shared memory
    for (int i = tid; i < head_size; i += blockDim.x) {
        Qi[i] = q[i];
    }
    
    // Initialize output
    for (int i = tid; i < head_size; i += blockDim.x) {
        Oi_local[i] = 0.0f;
    }
    __syncthreads();
    
    // Process in blocks of Bc positions
    int num_blocks = (seq_len + Bc - 1) / Bc;
    
    for (int block = 0; block < num_blocks; block++) {
        int start_pos = block * Bc;
        int end_pos = min(start_pos + Bc, seq_len);
        int actual_Bc = end_pos - start_pos;
        
        // Load K and V blocks
        for (int idx = tid; idx < actual_Bc * head_size; idx += blockDim.x) {
            int pos = idx / head_size;
            int dim = idx % head_size;
            int global_pos = start_pos + pos;
            
            Kj[pos * head_size + dim] = k_cache[global_pos * kv_stride + dim];
            Vj[pos * head_size + dim] = v_cache[global_pos * kv_stride + dim];
        }
        __syncthreads();
        
        // Compute attention scores: Sij = Qi * Kj^T
        for (int pos = tid; pos < actual_Bc; pos += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_size; d++) {
                score += Qi[d] * Kj[pos * head_size + d];
            }
            Sij[pos] = score * scale;  // Apply scaling
        }
        __syncthreads();
        
        // Compute softmax statistics
        if (tid == 0) {
            // Find max in this block
            float block_max = -FLT_MAX;
            for (int pos = 0; pos < actual_Bc; pos++) {
                block_max = fmaxf(block_max, Sij[pos]);
            }
            
            // Update global max
            float old_mi = stats[0];
            float new_mi = fmaxf(old_mi, block_max);
            
            // Compute probabilities and sum
            float block_sum = 0.0f;
            for (int pos = 0; pos < actual_Bc; pos++) {
                float prob = expf(Sij[pos] - new_mi);
                Pij[pos] = prob;
                block_sum += prob;
            }
            
            // Update global sum with rescaling
            float old_li = stats[1];
            float new_li = expf(old_mi - new_mi) * old_li + block_sum;
            
            // Store updated statistics
            stats[0] = new_mi;
            stats[1] = new_li;
            
            // Rescale existing output
            if (block > 0) {
                float rescale = expf(old_mi - new_mi);
                for (int d = 0; d < head_size; d++) {
                    Oi_local[d] *= rescale;
                }
            }
        }
        __syncthreads();
        
        // Add contribution from this block
        for (int d = tid; d < head_size; d += blockDim.x) {
            float contribution = 0.0f;
            for (int pos = 0; pos < actual_Bc; pos++) {
                contribution += Pij[pos] * Vj[pos * head_size + d];
            }
            Oi_local[d] += contribution;
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    if (tid == 0) {
        float final_norm = 1.0f / stats[1];  // 1 / li
        for (int d = 0; d < head_size; d++) {
            output[d] = Oi_local[d] * final_norm;
        }
    }
}


// ---------------------------------------------------------------------------- 
// Multi-head Flash Attention kernel

__global__ void multi_head_flashattention_kernel(
    const float* all_queries,  // [n_heads * head_size]
    const float* k_cache,      // Key cache with stride
    const float* v_cache,      // Value cache with stride  
    float* all_outputs,        // [n_heads * head_size]
    const int seq_len,         // Sequence length
    const int head_size,       // Head dimension
    const int n_heads,         // Number of query heads
    const int n_kv_heads,      // Number of KV heads
    const int kv_stride,       // Stride in KV cache
    const float scale,         // Attention scale
    const int Bc               // Block size (ADD THIS PARAMETER)
) {
    int head_id = blockIdx.x;
    if (head_id >= n_heads) return;
    
    // Calculate KV head (for Grouped Query Attention)
    int kv_head = head_id % n_kv_heads;
    
    // Get pointers for this head
    const float* q = all_queries + head_id * head_size;
    const float* k = k_cache + kv_head * head_size;
    const float* v = v_cache + kv_head * head_size;
    float* output = all_outputs + head_id * head_size;
    
    // Shared memory layout
    extern __shared__ float smem[];
    
    float* Kj = smem;                          // [Bc * head_size]
    float* Vj = Kj + Bc * head_size;           // [Bc * head_size]
    float* Qi = Vj + Bc * head_size;           // [head_size]
    float* Sij = Qi + head_size;               // [Bc]
    float* Pij = Sij + Bc;                     // [Bc]
    float* Oi_local = Pij + Bc;                // [head_size]
    float* stats = Oi_local + head_size;       // [6] for mi, li, etc.
    
    const int tid = threadIdx.x;
    
    // Initialize statistics
    if (tid == 0) {
        stats[0] = -FLT_MAX;  // mi (max)
        stats[1] = 0.0f;      // li (sum)
    }
    
    // Load query into shared memory
    for (int i = tid; i < head_size; i += blockDim.x) {
        Qi[i] = q[i];
    }
    
    // Initialize output
    for (int i = tid; i < head_size; i += blockDim.x) {
        Oi_local[i] = 0.0f;
    }
    __syncthreads();
    
    // Process in blocks of Bc positions
    int num_blocks = (seq_len + Bc - 1) / Bc;
    
    for (int block = 0; block < num_blocks; block++) {
        int start_pos = block * Bc;
        int end_pos = min(start_pos + Bc, seq_len);
        int actual_Bc = end_pos - start_pos;
        
        // Load K and V blocks
        for (int idx = tid; idx < actual_Bc * head_size; idx += blockDim.x) {
            int pos = idx / head_size;
            int dim = idx % head_size;
            int global_pos = start_pos + pos;
            
            Kj[pos * head_size + dim] = k[global_pos * kv_stride + dim];
            Vj[pos * head_size + dim] = v[global_pos * kv_stride + dim];
        }
        __syncthreads();
        
        // Compute attention scores: Sij = Qi * Kj^T
        for (int pos = tid; pos < actual_Bc; pos += blockDim.x) {
            float score = 0.0f;
            for (int d = 0; d < head_size; d++) {
                score += Qi[d] * Kj[pos * head_size + d];
            }
            Sij[pos] = score * scale;  // Apply scaling
        }
        __syncthreads();
        
        // Compute softmax statistics
        if (tid == 0) {
            // Find max in this block
            float block_max = -FLT_MAX;
            for (int pos = 0; pos < actual_Bc; pos++) {
                block_max = fmaxf(block_max, Sij[pos]);
            }
            
            // Update global max
            float old_mi = stats[0];
            float new_mi = fmaxf(old_mi, block_max);
            
            // Compute probabilities and sum
            float block_sum = 0.0f;
            for (int pos = 0; pos < actual_Bc; pos++) {
                float prob = expf(Sij[pos] - new_mi);
                Pij[pos] = prob;
                block_sum += prob;
            }
            
            // Update global sum with rescaling
            float old_li = stats[1];
            float new_li = expf(old_mi - new_mi) * old_li + block_sum;
            
            // Store updated statistics
            stats[0] = new_mi;
            stats[1] = new_li;
            
            // Rescale existing output
            if (block > 0) {
                float rescale = expf(old_mi - new_mi);
                for (int d = 0; d < head_size; d++) {
                    Oi_local[d] *= rescale;
                }
            }
        }
        __syncthreads();
        
        // Add contribution from this block
        for (int d = tid; d < head_size; d += blockDim.x) {
            float contribution = 0.0f;
            for (int pos = 0; pos < actual_Bc; pos++) {
                contribution += Pij[pos] * Vj[pos * head_size + d];
            }
            Oi_local[d] += contribution;
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    if (tid == 0) {
        float final_norm = 1.0f / stats[1];  // 1 / li
        for (int d = 0; d < head_size; d++) {
            output[d] = Oi_local[d] * final_norm;
        }
    }
}



// ----------------------------------------------------------------------------
// Wrapper function for fused QKV projection
void cuda_fused_qkv_projection(
    const float* input,     // Input vector [dim]
    const float* Wq,        // Query weights [dim x dim]
    const float* Wk,        // Key weights [dim x kv_dim] 
    const float* Wv,        // Value weights [dim x kv_dim]
    float* q_out,           // Query output [dim]
    float* k_out,           // Key output [kv_dim]
    float* v_out,           // Value output [kv_dim]
    const int dim,          // Input/Query dimension
    const int kv_dim,       // Key/Value dimension
    const float alpha,      // Scaling factor
    const float beta        // Bias factor
) {
    // Launch 3 blocks: one for Q, one for K, one for V
    dim3 gridDim(3);        // 3 blocks total
    dim3 blockDim(256);     // 256 threads per block
    
    // Shared memory size for input vector
    size_t smem_size = dim * sizeof(float);
    
    fused_qkv_kernel<<<gridDim, blockDim, smem_size>>>(
        input, Wq, Wk, Wv, q_out, k_out, v_out, 
        dim, kv_dim, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------
// Fused output projection + residual connection kernel
__global__ void fused_output_residual_kernel(
    const float* attention_out,  // Attention output [dim]
    const float* Wo,            // Output projection weights [dim x dim]
    float* residual_inout,      // Input/output for residual [dim] (modified in-place)
    const int dim,              // Dimension
    const float alpha           // Scaling factor
) {
    int tid = threadIdx.x;
    
    // Shared memory for attention output
    extern __shared__ float smem[];
    float* att_shared = smem;
    
    // Load attention output into shared memory cooperatively
    for (int i = tid; i < dim; i += blockDim.x) {
        att_shared[i] = attention_out[i];
    }
    __syncthreads();
    
    // Each thread computes one output element and adds residual
    for (int out_idx = tid; out_idx < dim; out_idx += blockDim.x) {
        float projection_result = 0.0f;
        
        // Compute projection: proj[out_idx] = sum(attention_out[i] * Wo[out_idx, i])
        for (int i = 0; i < dim; i++) {
            projection_result += att_shared[i] * Wo[out_idx * dim + i];
        }
        
        // Apply residual connection in-place: residual += alpha * projection
        residual_inout[out_idx] += alpha * projection_result;
    }
}


// ----------------------------------------------------------------------------

// Fused FFN W1+W3 projections kernel
__global__ void fused_ffn_w1w3_kernel(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate projection weights [dim x hidden_dim]
    const float* W3,        // Up projection weights [dim x hidden_dim]
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor (typically 0)
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input vector (reused by both projections)
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    // Load input vector into shared memory cooperatively
    for (int i = tid; i < dim; i += blockDim.x) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Block 0: Compute gate projection (W1)
    if (bid == 0) {
        // Each thread computes multiple output elements
        for (int out_idx = tid; out_idx < hidden_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Compute dot product: gate[out_idx] = sum(input[i] * W1[out_idx, i])
            // W1 is stored in column-major: W1[out_idx, i] = W1[out_idx * dim + i]
            for (int i = 0; i < dim; i++) {
                sum += input_shared[i] * W1[out_idx * dim + i];
            }
            
            gate_out[out_idx] = alpha * sum + beta * gate_out[out_idx];
        }
    }
    
    // Block 1: Compute up projection (W3)
    else if (bid == 1) {
        // Each thread computes multiple output elements
        for (int out_idx = tid; out_idx < hidden_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Compute dot product: up[out_idx] = sum(input[i] * W3[out_idx, i])
            // W3 is stored in column-major: W3[out_idx, i] = W3[out_idx * dim + i]
            for (int i = 0; i < dim; i++) {
                sum += input_shared[i] * W3[out_idx * dim + i];
            }
            
            up_out[out_idx] = alpha * sum + beta * up_out[out_idx];
        }
    }
}
// ----------------------------------------------------------------------------

// Alternative: Chunked version for very large input dimensions
__global__ void fused_ffn_w1w3_chunked_kernel(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate projection weights [dim x hidden_dim]
    const float* W3,        // Up projection weights [dim x hidden_dim]
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input chunks
    extern __shared__ float smem[];
    float* input_chunk = smem;
    
    const int chunk_size = min((int)blockDim.x, dim);
    
    // Block 0: Compute W1 projection
    if (bid == 0) {
        for (int out_idx = tid; out_idx < hidden_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Process input in chunks
            for (int chunk_start = 0; chunk_start < dim; chunk_start += chunk_size) {
                int chunk_end = min(chunk_start + chunk_size, dim);
                int chunk_len = chunk_end - chunk_start;
                
                // Load input chunk cooperatively
                for (int i = tid; i < chunk_len; i += blockDim.x) {
                    if (chunk_start + i < dim) {
                        input_chunk[i] = input[chunk_start + i];
                    }
                }
                __syncthreads();
                
                // Compute partial dot product
                for (int i = 0; i < chunk_len; i++) {
                    int global_i = chunk_start + i;
                    if (global_i < dim) {
                        sum += input_chunk[i] * W1[out_idx * dim + global_i];
                    }
                }
                __syncthreads();
            }
            
            gate_out[out_idx] = alpha * sum + beta * gate_out[out_idx];
        }
    }
    
    // Block 1: Compute W3 projection
    else if (bid == 1) {
        for (int out_idx = tid; out_idx < hidden_dim; out_idx += blockDim.x) {
            float sum = 0.0f;
            
            // Process input in chunks
            for (int chunk_start = 0; chunk_start < dim; chunk_start += chunk_size) {
                int chunk_end = min(chunk_start + chunk_size, dim);
                int chunk_len = chunk_end - chunk_start;
                
                // Load input chunk cooperatively
                for (int i = tid; i < chunk_len; i += blockDim.x) {
                    if (chunk_start + i < dim) {
                        input_chunk[i] = input[chunk_start + i];
                    }
                }
                __syncthreads();
                
                // Compute partial dot product
                for (int i = 0; i < chunk_len; i++) {
                    int global_i = chunk_start + i;
                    if (global_i < dim) {
                        sum += input_chunk[i] * W3[out_idx * dim + global_i];
                    }
                }
                __syncthreads();
            }
            
            up_out[out_idx] = alpha * sum + beta * up_out[out_idx];
        }
    }
}

// Chunked wrapper (for large dimensions)
void cuda_fused_ffn_w1w3_chunked(
    const float* input, const float* W1, const float* W3,
    float* gate_out, float* up_out,
    const int dim, const int hidden_dim, const float alpha, const float beta
) {
    dim3 gridDim(2);
    dim3 blockDim(256);
    
    // Use smaller shared memory for chunked processing
    size_t smem_size = blockDim.x * sizeof(float);
    
    fused_ffn_w1w3_chunked_kernel<<<gridDim, blockDim, smem_size>>>(
        input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Highly optimized version with better memory access patterns
__global__ void fused_ffn_w1w3_optimized_kernel(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate weights [dim x hidden_dim]
    const float* W3,        // Up weights [dim x hidden_dim] 
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input vector
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    // Cooperatively load input into shared memory
    for (int i = tid; i < dim; i += blockDim.x) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Each block handles one projection type
    const float* weight_matrix = (bid == 0) ? W1 : W3;
    float* output = (bid == 0) ? gate_out : up_out;
    
    // Each thread processes multiple outputs with vectorized operations
    const int outputs_per_thread = (hidden_dim + blockDim.x - 1) / blockDim.x;
    
    for (int t = 0; t < outputs_per_thread; t++) {
        int out_idx = tid + t * blockDim.x;
        if (out_idx < hidden_dim) {
            float sum = 0.0f;
            
            // Unroll inner loop for better performance
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                sum += input_shared[i    ] * weight_matrix[out_idx * dim + i    ];
                sum += input_shared[i + 1] * weight_matrix[out_idx * dim + i + 1];
                sum += input_shared[i + 2] * weight_matrix[out_idx * dim + i + 2];
                sum += input_shared[i + 3] * weight_matrix[out_idx * dim + i + 3];
            }
            
            // Handle remaining elements
            for (; i < dim; i++) {
                sum += input_shared[i] * weight_matrix[out_idx * dim + i];
            }
            
            output[out_idx] = alpha * sum + beta * output[out_idx];
        }
    }
}

// Optimized wrapper
void cuda_fused_ffn_w1w3_optimized(
    const float* input, const float* W1, const float* W3,
    float* gate_out, float* up_out,
    const int dim, const int hidden_dim, const float alpha, const float beta
) {
    dim3 gridDim(2);
    dim3 blockDim(256);
    
    size_t smem_size = dim * sizeof(float);
    
    if (smem_size <= 48 * 1024) {
        fused_ffn_w1w3_optimized_kernel<<<gridDim, blockDim, smem_size>>>(
            input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
        );
    } else {
        // Fall back to chunked version
        cuda_fused_ffn_w1w3_chunked(input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------
// Wrapper function for fused FFN W1+W3 projections
void cuda_fused_ffn_w1w3(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate weights [dim x hidden_dim]
    const float* W3,        // Up weights [dim x hidden_dim]
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor
) {
    // Launch 2 blocks: one for W1 (gate), one for W3 (up)
    dim3 gridDim(2);
    dim3 blockDim(256);
    
    // Shared memory size for input vector
    size_t smem_size = dim * sizeof(float);
    
    // Check if shared memory fits (for dim=4096, this is 16KB, well under 48KB limit)
    if (smem_size > 48 * 1024) {
        printf("Warning: Input dimension too large for shared memory, falling back to separate operations\n");
        cuda_sgemv(gate_out, W1, input, dim, hidden_dim, alpha, beta, true);
        cuda_sgemv(up_out, W3, input, dim, hidden_dim, alpha, beta, true);
        return;
    }
    
    fused_ffn_w1w3_kernel<<<gridDim, blockDim, smem_size>>>(
        input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------

// More aggressive FFN W1+W3 optimization using tiled approach
__global__ void improved_ffn_w1w3_kernel(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate weights [dim x hidden_dim]
    const float* W3,        // Up weights [dim x hidden_dim]
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor
) {
    // Use all blocks to process outputs, not just 2 blocks
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int total_blocks = gridDim.x;
    
    // Shared memory for input vector
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    // Load input cooperatively across all blocks
    for (int i = tid; i < dim; i += blockDim.x) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Calculate how many outputs each block processes
    int outputs_per_block = (hidden_dim + total_blocks - 1) / total_blocks;
    int output_start = bid * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, hidden_dim);
    
    // Each thread in the block processes multiple outputs
    for (int base_out = output_start; base_out < output_end; base_out += blockDim.x) {
        int out_idx = base_out + tid;
        
        if (out_idx < hidden_dim) {
            // Compute both W1 and W3 projections for this output
            float sum_w1 = 0.0f;
            float sum_w3 = 0.0f;
            
            // Vectorized inner loop with unrolling
            int i = 0;
            for (; i + 3 < dim; i += 4) {
                float in0 = input_shared[i    ];
                float in1 = input_shared[i + 1];
                float in2 = input_shared[i + 2];
                float in3 = input_shared[i + 3];
                
                // W1 computation
                sum_w1 += in0 * W1[out_idx * dim + i    ];
                sum_w1 += in1 * W1[out_idx * dim + i + 1];
                sum_w1 += in2 * W1[out_idx * dim + i + 2];
                sum_w1 += in3 * W1[out_idx * dim + i + 3];
                
                // W3 computation
                sum_w3 += in0 * W3[out_idx * dim + i    ];
                sum_w3 += in1 * W3[out_idx * dim + i + 1];
                sum_w3 += in2 * W3[out_idx * dim + i + 2];
                sum_w3 += in3 * W3[out_idx * dim + i + 3];
            }
            
            // Handle remaining elements
            for (; i < dim; i++) {
                float input_val = input_shared[i];
                sum_w1 += input_val * W1[out_idx * dim + i];
                sum_w3 += input_val * W3[out_idx * dim + i];
            }
            
            // Write results
            gate_out[out_idx] = alpha * sum_w1 + beta * gate_out[out_idx];
            up_out[out_idx] = alpha * sum_w3 + beta * up_out[out_idx];
        }
    }
}

// ----------------------------------------------------------------------------
// Complete memory-optimized FFN W1+W3 kernel
__global__ void memory_optimized_ffn_w1w3_kernel(
    const float* input,     // Input vector [dim]
    const float* W1,        // Gate weights [dim x hidden_dim]
    const float* W3,        // Up weights [dim x hidden_dim]
    float* gate_out,        // Gate output [hidden_dim]
    float* up_out,          // Up output [hidden_dim]
    const int dim,          // Input dimension
    const int hidden_dim,   // Hidden dimension
    const float alpha,      // Scaling factor
    const float beta        // Beta factor
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for input vector
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    // Load input vector cooperatively
    for (int i = tid; i < dim; i += blockDim.x) {
        input_shared[i] = input[i];
    }
    __syncthreads();
    
    // Calculate output range for this block
    int outputs_per_block = (hidden_dim + gridDim.x - 1) / gridDim.x;
    int output_start = bid * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, hidden_dim);
    
    // Each thread processes multiple outputs
    for (int out_idx = output_start + tid; out_idx < output_end; out_idx += blockDim.x) {
        float sum_w1 = 0.0f;
        float sum_w3 = 0.0f;
        
        // Process input in chunks for better cache utilization
        const int CHUNK_SIZE = 128;  // Tuned for L1 cache
        for (int chunk = 0; chunk < dim; chunk += CHUNK_SIZE) {
            int chunk_end = min(chunk + CHUNK_SIZE, dim);
            
            // Process this chunk with vectorized operations
            int i = chunk;
            for (; i + 3 < chunk_end; i += 4) {
                float in0 = input_shared[i    ];
                float in1 = input_shared[i + 1];
                float in2 = input_shared[i + 2];
                float in3 = input_shared[i + 3];
                
                // W1 computation (unrolled)
                sum_w1 += in0 * W1[out_idx * dim + i    ];
                sum_w1 += in1 * W1[out_idx * dim + i + 1];
                sum_w1 += in2 * W1[out_idx * dim + i + 2];
                sum_w1 += in3 * W1[out_idx * dim + i + 3];
                
                // W3 computation (unrolled)
                sum_w3 += in0 * W3[out_idx * dim + i    ];
                sum_w3 += in1 * W3[out_idx * dim + i + 1];
                sum_w3 += in2 * W3[out_idx * dim + i + 2];
                sum_w3 += in3 * W3[out_idx * dim + i + 3];
            }
            
            // Handle remaining elements in chunk
            for (; i < chunk_end; i++) {
                float input_val = input_shared[i];
                sum_w1 += input_val * W1[out_idx * dim + i];
                sum_w3 += input_val * W3[out_idx * dim + i];
            }
        }
        
        // Write results
        gate_out[out_idx] = alpha * sum_w1 + beta * gate_out[out_idx];
        up_out[out_idx] = alpha * sum_w3 + beta * up_out[out_idx];
    }
}

// ----------------------------------------------------------------------------


// Optimized for A100 
// Complete memory-optimized FFN W1+W3 kernel

__global__ void ffn_w1w3_a100_ultra_optimized(
    const float* __restrict__ input,
    const float* __restrict__ W1,
    const float* __restrict__ W3,
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int dim,
    const int hidden_dim,
    const float alpha,
    const float beta
) {
    extern __shared__ float smem[];
    float* input_shared = smem;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_blocks = gridDim.x;
    const int block_size = blockDim.x;
    
    // Load input with vectorization
    const int vec_size = 4;
    const int vectorized_dim = (dim + vec_size - 1) / vec_size;
    
    for (int i = tid; i < vectorized_dim; i += block_size) {
        int base_idx = i * vec_size;
        if (base_idx + 3 < dim) {
            float4 input_vec = *reinterpret_cast<const float4*>(&input[base_idx]);
            *reinterpret_cast<float4*>(&input_shared[base_idx]) = input_vec;
        } else {
            for (int j = 0; j < vec_size && base_idx + j < dim; j++) {
                input_shared[base_idx + j] = input[base_idx + j];
            }
        }
    }
    __syncthreads();
    
    // Determine which projection this block handles
    int projection_type = bid % 2;  // 0=W1, 1=W3
    int block_in_type = bid / 2;
    
    const float* weight_matrix = (projection_type == 0) ? W1 : W3;
    float* output = (projection_type == 0) ? gate_out : up_out;
    
    // Distribute work across blocks
    int outputs_per_type_block = (hidden_dim + (total_blocks/2) - 1) / (total_blocks/2);
    int output_start = block_in_type * outputs_per_type_block;
    int output_end = min(output_start + outputs_per_type_block, hidden_dim);
    
    for (int out_idx = output_start + tid; out_idx < output_end; out_idx += block_size) {
        float sum = 0.0f;
        
        // Ultra-aggressive 64-way unrolling for A100
        int i = 0;
        for (; i + 63 < dim; i += 64) {
            #pragma unroll 64
            for (int j = 0; j < 64; j++) {
                sum += input_shared[i + j] * weight_matrix[out_idx * dim + i + j];
            }
        }
        
        for (; i < dim; i++) {
            sum += input_shared[i] * weight_matrix[out_idx * dim + i];
        }
        
        output[out_idx] = alpha * sum + beta * output[out_idx];
    }
}

// ----------------------------------------------------------------------------


// Improved wrapper with better parallelization
void cuda_improved_ffn_w1w3(
    const float* input, const float* W1, const float* W3,
    float* gate_out, float* up_out,
    const int dim, const int hidden_dim, const float alpha, const float beta
) {
    // Use more blocks for better parallelization
    int max_blocks = min(64, (hidden_dim + 127) / 128);  // More blocks than before
    dim3 gridDim(max_blocks);
    dim3 blockDim(256);
    
    size_t smem_size = dim * sizeof(float);
    
    if (smem_size <= 48 * 1024) {
        improved_ffn_w1w3_kernel<<<gridDim, blockDim, smem_size>>>(
            input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
        );
    } else {
        // Fall back to chunked version for very large dimensions  
        cuda_fused_ffn_w1w3_chunked(input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------

// Memory-optimized wrapper
void cuda_memory_optimized_ffn_w1w3(
    const float* input, const float* W1, const float* W3,
    float* gate_out, float* up_out,
    const int dim, const int hidden_dim, const float alpha, const float beta
) {
    // Use many blocks for better occupancy
    int target_outputs_per_block = 64;  // Tune this value
    int num_blocks = (hidden_dim + target_outputs_per_block - 1) / target_outputs_per_block;
    num_blocks = min(num_blocks, 108);  // Cap at 108 blocks. earlier 128
    
    dim3 gridDim(num_blocks);
    dim3 blockDim(1024);              // For A100, use larger blocks, started with 256 threads
    
    size_t smem_size = dim * sizeof(float);
    
    if (smem_size <= 48 * 1024) {
        memory_optimized_ffn_w1w3_kernel<<<gridDim, blockDim, smem_size>>>(
            input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
        );
    } else {
        // Fall back to chunked version
        printf("Warning: Using chunked version due to large dimension\n");
        cuda_fused_ffn_w1w3_chunked(input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}


void cuda_ffn_w1w3_ultra_optimized(
    const float* input, const float* W1, const float* W3,
    float* gate_out, float* up_out,
    const int dim, const int hidden_dim, 
    const float alpha, const float beta
) {
    GPUConfig config = detect_gpu_config();
    
    if (config.is_a100) {
        // Use many blocks to saturate A100
        int blocks_per_projection = min(config.sm_count / 2, (hidden_dim + 1023) / 1024);
        int total_blocks = blocks_per_projection * 2;
        
        dim3 gridDim(total_blocks);
        dim3 blockDim(1024);
        size_t smem_size = dim * sizeof(float);
        
        ffn_w1w3_a100_ultra_optimized<<<gridDim, blockDim, smem_size>>>(
            input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta
        );
    } else {
        // Fallback to existing implementation
        cuda_memory_optimized_ffn_w1w3(input, W1, W3, gate_out, up_out, dim, hidden_dim, alpha, beta);
    }
}

// ----------------------------------------------------------------------------

// Fused SwiGLU + W2 projection kernel
__global__ void fused_swiglu_w2_kernel(
    const float* gate_input,    // Gate projection output [hidden_dim]
    const float* up_input,      // Up projection output [hidden_dim]
    const float* W2,            // W2 weights [hidden_dim x dim]
    float* output,              // Final output [dim]
    const int hidden_dim,       // Hidden dimension (11008)
    const int dim,              // Output dimension (4096)
    const float alpha,          // Scaling factor
    const float beta            // Beta factor
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for SwiGLU results
    extern __shared__ float smem[];
    float* swiglu_shared = smem;
    
    // Step 1: Compute SwiGLU cooperatively and store in shared memory
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float gate_val = gate_input[i];
        float up_val = up_input[i];
        
        // SwiGLU = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
        float silu_gate = gate_val / (1.0f + expf(-gate_val));
        swiglu_shared[i] = silu_gate * up_val;
    }
    __syncthreads();
    
    // Step 2: Compute W2 projection using SwiGLU results
    // Each thread block handles multiple output elements
    int outputs_per_block = (dim + gridDim.x - 1) / gridDim.x;
    int output_start = bid * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, dim);
    
    for (int out_idx = output_start + tid; out_idx < output_end; out_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Compute dot product: output[out_idx] = sum(swiglu_result[i] * W2[out_idx, i])
        // W2 is stored in column-major: W2[out_idx, i] = W2[out_idx * hidden_dim + i]
        
        // Unroll for better performance
        int i = 0;
        for (; i + 3 < hidden_dim; i += 4) {
            sum += swiglu_shared[i    ] * W2[out_idx * hidden_dim + i    ];
            sum += swiglu_shared[i + 1] * W2[out_idx * hidden_dim + i + 1];
            sum += swiglu_shared[i + 2] * W2[out_idx * hidden_dim + i + 2];
            sum += swiglu_shared[i + 3] * W2[out_idx * hidden_dim + i + 3];
        }
        
        // Handle remaining elements
        for (; i < hidden_dim; i++) {
            sum += swiglu_shared[i] * W2[out_idx * hidden_dim + i];
        }
        
        output[out_idx] = alpha * sum + beta * output[out_idx];
    }
}


// Chunked version for large hidden dimensions
__global__ void fused_swiglu_w2_chunked_kernel(
    const float* gate_input, const float* up_input, const float* W2, float* output,
    const int hidden_dim, const int dim, const float alpha, const float beta
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Shared memory for SwiGLU chunk
    extern __shared__ float smem[];
    float* swiglu_chunk = smem;
    
    const int chunk_size = blockDim.x;
    
    // Calculate output range for this block
    int outputs_per_block = (dim + gridDim.x - 1) / gridDim.x;
    int output_start = bid * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, dim);
    
    for (int out_idx = output_start + tid; out_idx < output_end; out_idx += blockDim.x) {
        float sum = 0.0f;
        
        // Process hidden dimension in chunks
        for (int chunk_start = 0; chunk_start < hidden_dim; chunk_start += chunk_size) {
            int chunk_end = min(chunk_start + chunk_size, hidden_dim);
            int chunk_len = chunk_end - chunk_start;
            
            // Compute SwiGLU for this chunk
            for (int i = tid; i < chunk_len; i += blockDim.x) {
                int global_i = chunk_start + i;
                if (global_i < hidden_dim) {
                    float gate_val = gate_input[global_i];
                    float up_val = up_input[global_i];
                    float silu_gate = gate_val / (1.0f + expf(-gate_val));
                    swiglu_chunk[i] = silu_gate * up_val;
                }
            }
            __syncthreads();
            
            // Compute partial dot product
            for (int i = 0; i < chunk_len; i++) {
                int global_i = chunk_start + i;
                if (global_i < hidden_dim) {
                    sum += swiglu_chunk[i] * W2[out_idx * hidden_dim + global_i];
                }
            }
            __syncthreads();
        }
        
        output[out_idx] = alpha * sum + beta * output[out_idx];
    }
}

// ----------------------------------------------------------------------------
// A100-Optimized RMSNorm with vectorized memory access and warp primitives



// Optimized RMSNorm kernel for A100
__global__ void rmsnorm_kernel_a100_optimized(
    float* __restrict__ output, 
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    int size, 
    float eps
) {
    // Use A100's 128KB L1 cache efficiently
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Vectorized loading for better memory bandwidth
    const int vec_size = 4;  // Load 4 floats at once (128-bit loads)
    const int vectorized_size = (size + vec_size - 1) / vec_size;
    
    // Phase 1: Compute sum of squares with vectorized loads
    float local_sum_sq = 0.0f;
    
    // Process 4 elements at a time for better memory bandwidth
    for (int i = tid; i < vectorized_size; i += block_size) {
        int base_idx = i * vec_size;
        
        // Vectorized load (4 floats at once)
        float4 input_vec = {0.0f, 0.0f, 0.0f, 0.0f};
        if (base_idx < size) {
            // Coalesced 128-bit load
            if (base_idx + 3 < size) {
                input_vec = *reinterpret_cast<const float4*>(&input[base_idx]);
            } else {
                // Handle remaining elements
                for (int j = 0; j < vec_size && base_idx + j < size; j++) {
                    reinterpret_cast<float*>(&input_vec)[j] = input[base_idx + j];
                }
            }
        }
        
        // Accumulate squares
        local_sum_sq += input_vec.x * input_vec.x;
        local_sum_sq += input_vec.y * input_vec.y;
        local_sum_sq += input_vec.z * input_vec.z;
        local_sum_sq += input_vec.w * input_vec.w;
    }
    
    // Phase 2: Warp-level reduction using A100's shuffle instructions
    // A100 has very fast warp shuffle - much better than shared memory
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
    }
    
    // Phase 3: Block-level reduction
    __shared__ float warp_sums[32];  // Max 32 warps per block
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        float sum = (lane_id < (block_size + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            shared_mem[0] = rsqrtf(sum / size + eps);  // Store RMS scale
        }
    }
    __syncthreads();
    
    float rms_scale = shared_mem[0];
    
    // Phase 4: Apply normalization with vectorized stores
    for (int i = tid; i < vectorized_size; i += block_size) {
        int base_idx = i * vec_size;
        
        if (base_idx < size) {
            // Vectorized load
            float4 input_vec = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 weight_vec = {0.0f, 0.0f, 0.0f, 0.0f};
            
            if (base_idx + 3 < size) {
                input_vec = *reinterpret_cast<const float4*>(&input[base_idx]);
                weight_vec = *reinterpret_cast<const float4*>(&weight[base_idx]);
            } else {
                // Handle remaining elements
                for (int j = 0; j < vec_size && base_idx + j < size; j++) {
                    reinterpret_cast<float*>(&input_vec)[j] = input[base_idx + j];
                    reinterpret_cast<float*>(&weight_vec)[j] = weight[base_idx + j];
                }
            }
            
            // Apply normalization
            float4 output_vec;
            output_vec.x = input_vec.x * rms_scale * weight_vec.x;
            output_vec.y = input_vec.y * rms_scale * weight_vec.y;
            output_vec.z = input_vec.z * rms_scale * weight_vec.z;
            output_vec.w = input_vec.w * rms_scale * weight_vec.w;
            
            // Vectorized store
            if (base_idx + 3 < size) {
                *reinterpret_cast<float4*>(&output[base_idx]) = output_vec;
            } else {
                // Handle remaining elements
                for (int j = 0; j < vec_size && base_idx + j < size; j++) {
                    output[base_idx + j] = reinterpret_cast<float*>(&output_vec)[j];
                }
            }
        }
    }
}



// RTX 2070 Super optimized kernel
__global__ void rmsnorm_kernel_rtx2070_optimized(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int size,
    float eps
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // RTX 2070 Super: Use 2-element vectorization (conservative)
    float local_sum_sq = 0.0f;
    
    // Process 2 elements at a time (64-bit loads - safe for RTX 2070 Super)
    for (int i = tid * 2; i < size; i += block_size * 2) {
        if (i < size) {
            float val1 = input[i];
            local_sum_sq += val1 * val1;
        }
        if (i + 1 < size) {
            float val2 = input[i + 1];
            local_sum_sq += val2 * val2;
        }
    }
    
    // Warp-level reduction (supported on RTX 2070 Super)
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
    }
    
    // Block-level reduction using shared memory
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = (lane_id < (block_size + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            shared_mem[0] = rsqrtf(sum / size + eps);
        }
    }
    __syncthreads();
    
    float rms_scale = shared_mem[0];
    
    // Apply normalization (2-element vectorization)
    for (int i = tid * 2; i < size; i += block_size * 2) {
        if (i < size) {
            output[i] = input[i] * rms_scale * weight[i];
        }
        if (i + 1 < size) {
            output[i + 1] = input[i + 1] * rms_scale * weight[i + 1];
        }
    }
}

// ----------------------------------------------------------------------------

// A100-optimized output projection + residual (similar approach to QKV)
__global__ void output_projection_a100_optimized(
    const float* __restrict__ attention_out,  // [dim]
    const float* __restrict__ Wo,             // [dim x dim]
    float* __restrict__ residual_inout,       // [dim] (modified in-place)
    const int dim,
    const float alpha
) {
    extern __shared__ float smem[];
    float* attention_shared = smem;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_blocks = gridDim.x;
    const int block_size = blockDim.x;
    
    // Cooperatively load attention output with vectorization
    const int vec_size = 4;
    const int vectorized_dim = (dim + vec_size - 1) / vec_size;
    
    for (int i = tid; i < vectorized_dim; i += block_size) {
        int base_idx = i * vec_size;
        if (base_idx + 3 < dim) {
            float4 att_vec = *reinterpret_cast<const float4*>(&attention_out[base_idx]);
            *reinterpret_cast<float4*>(&attention_shared[base_idx]) = att_vec;
        } else {
            for (int j = 0; j < vec_size && base_idx + j < dim; j++) {
                attention_shared[base_idx + j] = attention_out[base_idx + j];
            }
        }
    }
    __syncthreads();
    
    // Distribute output computation across blocks for A100's 108 SMs
    int outputs_per_block = (dim + total_blocks - 1) / total_blocks;
    int output_start = bid * outputs_per_block;
    int output_end = min(output_start + outputs_per_block, dim);
    
    for (int out_idx = output_start + tid; out_idx < output_end; out_idx += block_size) {
        float projection_result = 0.0f;
        
        // Aggressive 32-way unrolling for A100
        int i = 0;
        for (; i + 31 < dim; i += 32) {
            #pragma unroll 32
            for (int j = 0; j < 32; j++) {
                projection_result += attention_shared[i + j] * Wo[out_idx * dim + i + j];
            }
        }
        
        // Handle remaining elements  
        for (; i < dim; i++) {
            projection_result += attention_shared[i] * Wo[out_idx * dim + i];
        }
        
        // Apply residual connection in-place
        residual_inout[out_idx] += alpha * projection_result;
    }
}

// RTX 2070 Super version (conservative)
__global__ void output_projection_rtx2070_optimized(
    const float* __restrict__ attention_out,
    const float* __restrict__ Wo,
    float* __restrict__ residual_inout,
    const int dim,
    const float alpha
) {
    extern __shared__ float smem[];
    float* attention_shared = smem;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Load attention output (2-element vectorization for RTX 2070 Super)
    for (int i = tid * 2; i < dim; i += block_size * 2) {
        if (i < dim) attention_shared[i] = attention_out[i];
        if (i + 1 < dim) attention_shared[i + 1] = attention_out[i + 1];
    }
    __syncthreads();
    
    // Compute output projection with 4-way unrolling
    for (int out_idx = tid; out_idx < dim; out_idx += block_size) {
        float projection_result = 0.0f;
        
        int i = 0;
        for (; i + 3 < dim; i += 4) {
            projection_result += attention_shared[i+0] * Wo[out_idx * dim + i+0];
            projection_result += attention_shared[i+1] * Wo[out_idx * dim + i+1]; 
            projection_result += attention_shared[i+2] * Wo[out_idx * dim + i+2];
            projection_result += attention_shared[i+3] * Wo[out_idx * dim + i+3];
        }
        
        for (; i < dim; i++) {
            projection_result += attention_shared[i] * Wo[out_idx * dim + i];
        }
        
        residual_inout[out_idx] += alpha * projection_result;
    }
}

// ----------------------------------------------------------------------------

void cuda_fused_swiglu_w2_chunked(
    const float* gate_input, const float* up_input, const float* W2, float* output,
    const int hidden_dim, const int dim, const float alpha, const float beta
) {
    int max_blocks = min(32, (dim + 255) / 256);
    dim3 gridDim(max_blocks);
    dim3 blockDim(256);
    
    size_t smem_size = blockDim.x * sizeof(float);
    
    fused_swiglu_w2_chunked_kernel<<<gridDim, blockDim, smem_size>>>(
        gate_input, up_input, W2, output, hidden_dim, dim, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------

// Wrapper function
void cuda_fused_swiglu_w2_projection(
    const float* gate_input,    // Gate projection output [hidden_dim]
    const float* up_input,      // Up projection output [hidden_dim]  
    const float* W2,            // W2 weights [hidden_dim x dim]
    float* output,              // Final output [dim]
    const int hidden_dim,       // Hidden dimension
    const int dim,              // Output dimension
    const float alpha,          // Scaling factor
    const float beta            // Beta factor
) {
    // Calculate grid/block dimensions
    int max_blocks = min(32, (dim + 255) / 256);  // Limit blocks for better occupancy
    dim3 gridDim(max_blocks);
    dim3 blockDim(256);
    
    // Shared memory for SwiGLU results
    size_t smem_size = hidden_dim * sizeof(float);
    
    // Check if shared memory fits
    if (smem_size > 48 * 1024) {
        printf("Warning: Hidden dimension too large for shared memory, using chunked version\n");
        cuda_fused_swiglu_w2_chunked(gate_input, up_input, W2, output, hidden_dim, dim, alpha, beta);
        return;
    }
    
    fused_swiglu_w2_kernel<<<gridDim, blockDim, smem_size>>>(
        gate_input, up_input, W2, output, hidden_dim, dim, alpha, beta
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------------------------------------------------------------

// Wrapper function
void cuda_fused_output_residual(
    const float* attention_out,  // Attention output [dim]
    const float* Wo,            // Output weights [dim x dim]
    float* residual_inout,      // Residual input/output [dim]
    const int dim,              // Dimension
    const float alpha           // Scaling factor
) {
    dim3 blockDim(256);
    dim3 gridDim(1);           // Single block since we need cooperation
    
    // Shared memory for attention output
    size_t smem_size = dim * sizeof(float);
    
    // Check if shared memory fits
    if (smem_size > 48 * 1024) {
        // Fall back to separate operations for very large dimensions
        printf("Warning: Dimension too large for fused kernel, using separate operations\n");
        
        // Temporary buffer for projection result
        float* temp_projection;
        cudaMalloc(&temp_projection, dim * sizeof(float));
        
        // Separate projection and residual
        cuda_sgemv(temp_projection, Wo, attention_out, dim, dim, alpha, 0.0f, true);
        cuda_saxpy(residual_inout, temp_projection, 1.0f, dim);
        
        cudaFree(temp_projection);
        return;
    }
    
    fused_output_residual_kernel<<<gridDim, blockDim, smem_size>>>(
        attention_out, Wo, residual_inout, dim, alpha
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}


// v2 of cuda_fused_output_residual for fused output projection adaptive to A100 and RTX 2070 Super
// Adaptive wrapper for output projection
void cuda_output_projection_adaptive_optimized(
    const float* attention_out,
    const float* Wo,
    float* residual_inout,
    const int dim,
    const float alpha
) {
    GPUConfig config = detect_gpu_config();
    size_t smem_size = dim * sizeof(float);
    
    if (config.is_a100) {
        // A100: Use all SMs for maximum parallelization
        int num_blocks = min(config.sm_count, (dim + 1023) / 1024);
        
        dim3 gridDim(num_blocks);
        dim3 blockDim(1024);
        
        // printf("A100 Output Projection: %d blocks, 1024 threads/block\n", num_blocks);
        
        output_projection_a100_optimized<<<gridDim, blockDim, smem_size>>>(
            attention_out, Wo, residual_inout, dim, alpha
        );
    }
    else if (config.is_rtx20_series) {
        // RTX 2070 Super: Conservative single block
        dim3 gridDim(1);
        dim3 blockDim(256);
        smem_size = min(smem_size, (size_t)(40 * 1024));
        
        output_projection_rtx2070_optimized<<<gridDim, blockDim, smem_size>>>(
            attention_out, Wo, residual_inout, dim, alpha
        );
    }
    else {
        // Fallback
        dim3 gridDim(1);
        dim3 blockDim(256);
        smem_size = min(smem_size, config.shared_mem_per_block / 2);
        
        output_projection_rtx2070_optimized<<<gridDim, blockDim, smem_size>>>(
            attention_out, Wo, residual_inout, dim, alpha
        );
    }
}

// ----------------------------------------------------------------------------

// A100-optimized configuration
int get_optimal_block_size_a100(int head_size, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    // A100 specific optimizations
    if (prop.major == 8 && prop.minor == 0) {  // A100
        // A100 has 164KB shared memory per block
        int max_shared = prop.sharedMemPerBlock;
        int usable_shared = (max_shared * 85) / 100;  // Use 85% for safety
        
        // printf("A100 detected: Using %d KB shared memory\n", usable_shared / 1024);
        return usable_shared;
    }
    
    // Default for other GPUs
    return 48 * 1024;
}

// ----------------------------------------------------------------------------

void cuda_rmsnorm(float* d_output, float* d_input, float* d_weight, int size) {
    // Note: 'layer' is not used in this simplified version, but could be used for layer-specific weights
    // 'pos' is used to print debug information if needed
#if USE_CUDA
    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    
    rmsnorm_kernel<<<gridDim, blockDim>>>(d_output, d_input, d_weight, size, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

#endif
}


// ----------------------------------------------------------------------------



void cuda_softmax(float* d_output, float* d_input, int size) {
#if USE_CUDA
    dim3 blockDim(256);
    dim3 gridDim(1);  // Single block for softmax
    
    softmax_kernel<<<gridDim, blockDim>>>(d_output, d_input, size);
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

// ----------------------------------------------------------------------------

void cuda_elementwise_add(float* d_a, float* d_b, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    elementwise_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, size);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Host functions 

// Add this profiling to your forward_gpu function to find real bottlenecks
void profile_7b_bottlenecks(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float scale = 1.0f / sqrtf((float)head_size);
    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    float stage1_ms = 0.0f;
    float stage2_ms = 0.0f;
    float stage3_ms = 0.0f;
    float stage4_ms = 0.0f;
    float stage5_ms = 0.0f;
    float stage6_ms = 0.0f;
    float stage7_ms = 0.0f;
    float stage8_ms = 0.0f;
    
    printf("\n=== MODEL BOTTLENECK ANALYSIS (Layer 0, pos=%d) ===\n", pos);
    
    // Copy token embedding
    float* token_embedding = w->token_embedding_table + token * dim;
    CUDA_CHECK(cudaMemcpy(s->d_x, token_embedding, dim * sizeof(float), cudaMemcpyHostToDevice));
    
    int l = 0; // Profile just first layer
    
    // 1. RMSNorm (what you optimized)
    cudaEventRecord(start);
    float* layer_rms_att_weight = w->d_rms_att_weight + l * dim;
    cuda_rmsnorm_adaptive(s->d_xb, s->d_x, layer_rms_att_weight, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage1_ms = elapsed_time;
    printf("1. RMSNorm (attention):        %.3f ms \n", elapsed_time);
    float total_time = elapsed_time;
    
    // 2. QKV Projections (MAJOR BOTTLENECK)
    cudaEventRecord(start);
    float* layer_wq = w->d_wq + l * dim * dim;
    float* layer_wk = w->d_wk + l * dim * kv_dim;  
    float* layer_wv = w->d_wv + l * dim * kv_dim;
    
    // cuda_fused_qkv_projection(
    //     s->d_xb, layer_wq, layer_wk, layer_wv,
    //     s->d_q, s->d_k, s->d_v,
    //     dim, kv_dim, alpha, beta
    // );

    // cuda_qkv_adaptive_optimized(
    //     s->d_xb, layer_wq, layer_wk, layer_wv,
    //     s->d_q, s->d_k, s->d_v,
    //     dim, kv_dim, alpha, beta
    // );

    cuda_matmul_2d_tiled(s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
    cuda_matmul_2d_tiled(s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
    cuda_matmul_2d_tiled(s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);

    // cuda_matmul_2d_vectorized  (s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
    // cuda_matmul_2d_vectorized  (s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
    // cuda_matmul_2d_vectorized  (s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);


    // Q = xb * Wq^T
    // const float alpha = 1.0f, beta = 0.0f;
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, 
    //         layer_wq, dim, s->d_xb, 1, &beta, s->d_q, 1);
    
    // // K = xb * Wk^T  
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, kv_dim, &alpha,
    //         layer_wk, dim, s->d_xb, 1, &beta, s->d_k, 1);
    
    // // V = xb * Wv^T
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, kv_dim, &alpha,
    //         layer_wv, dim, s->d_xb, 1, &beta, s->d_v, 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage2_ms = elapsed_time;
    printf("2. QKV Projections:            %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 3. RoPE + KV Cache
    cudaEventRecord(start);
    int total_elements = 1 * 1 * p->n_heads * head_size;
    dim3 blockDim(256);
    dim3 gridDim((total_elements / 2 + blockDim.x - 1) / blockDim.x);
    rope_kernel<<<gridDim, blockDim>>>(s->d_q, s->d_k, 1, 1, p->n_heads, head_size, pos);
    // CUDA_CHECK(cudaDeviceSynchronize());

    int loff = l * p->seq_len * kv_dim;
    float* d_key_cache_row = s->d_key_cache + loff + pos * kv_dim;
    float* d_value_cache_row = s->d_value_cache + loff + pos * kv_dim;

    CUDA_CHECK(cudaMemcpy(d_key_cache_row, s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_cache_row, s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage3_ms = elapsed_time;
    printf("3. RoPE + KV Cache:            %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 4. FlashAttention
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemset(s->d_xb, 0, dim * sizeof(float)));
    const int M = 164 * 1024;  // A100 optimized
    const int Bc = min(M / (4 * head_size), pos + 1);
    const int actual_Bc = min(Bc, 256);  // A100 optimized
    
    size_t smem_size = sizeof(float) * (
        actual_Bc * head_size + actual_Bc * head_size + head_size +
        actual_Bc + actual_Bc + head_size + 6
    );
    
    dim3 grid_fa(p->n_heads);
    dim3 block_fa(256);
    float* d_key_cache_layer = s->d_key_cache + loff;
    float* d_value_cache_layer = s->d_value_cache + loff;
    
    multi_head_flashattention_kernel<<<grid_fa, block_fa, smem_size>>>(
        s->d_q, d_key_cache_layer, d_value_cache_layer, s->d_xb,
        pos + 1, head_size, p->n_heads, p->n_kv_heads,
        p->n_kv_heads * head_size, scale, actual_Bc
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage4_ms = elapsed_time;
    printf("4. FlashAttention:             %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 5. Output Projection 
    cudaEventRecord(start);
    float* layer_wo = w->d_wo + l * dim * dim;
    // cuda_fused_output_residual(s->d_xb, layer_wo, s->d_x, dim, alpha);
    // cuda_output_projection_adaptive_optimized(
    //         s->d_xb,     // Attention output
    //         layer_wo,    // Output projection weights
    //         s->d_x,      // Input/output (in-place residual)
    //         dim, alpha
    //     );

    // CUDA_CHECK(cudaMemcpy(s->d_xb, s->xb, dim * sizeof(float), cudaMemcpyHostToDevice));
    cuda_matmul_2d_tiled(s->d_xb2, s->d_xb, layer_wo, dim, dim);
    cuda_elementwise_add(s->d_x, s->d_xb2, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage5_ms = elapsed_time;
    printf("5. Output Proj + Residual:     %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 6. FFN RMSNorm
    cudaEventRecord(start);
    float* layer_rms_ffn_weight = w->d_rms_ffn_weight + l * dim;
    cuda_rmsnorm_adaptive(s->d_xb, s->d_x, layer_rms_ffn_weight, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage6_ms = elapsed_time;
    printf("6. RMSNorm (FFN):              %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 7. FFN W1+W3 (BIGGEST BOTTLENECK)
    cudaEventRecord(start);
    float* layer_w1 = w->d_w1 + l * dim * hidden_dim;
    float* layer_w3 = w->d_w3 + l * dim * hidden_dim;

    // Match forward_gpu strict-isolation path: explicit W1 and W3 tiled matmuls.
    cuda_matmul_2d_tiled(s->d_hb, s->d_xb, layer_w1, dim, hidden_dim);
    cuda_matmul_2d_tiled(s->d_hb2, s->d_xb, layer_w3, dim, hidden_dim);

    // Cublas implementation for W1 and W3
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, hidden_dim, &alpha,
    //                layer_w1, dim, s->d_xb, 1, &beta, s->d_hb, 1);
    // cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, hidden_dim, &alpha, 
    //             layer_w3, dim, s->d_xb, 1, &beta, s->d_hb2, 1);

    // cuda_matmul(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
    // cuda_matmul(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);

    // cuda_matmul_2d_tiled(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
    // cuda_matmul_2d_tiled(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage7_ms = elapsed_time;
    printf("7. FFN W1+W3 :                 %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 8. SwiGLU + W2
    cudaEventRecord(start);
    float* layer_w2 = w->d_w2 + l * hidden_dim * dim;
    
    // cuda_fused_swiglu_w2_projection(
    //     s->d_hb, s->d_hb2, layer_w2, s->d_xb,
    //     hidden_dim, dim, alpha, beta
    // );
    
    // cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);
    // cuda_matmul(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
    // cuda_elementwise_add(s->d_x, s->d_xb, dim);


    // cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);
    // cuda_matmul_2d_tiled(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
    // cuda_elementwise_add(s->d_x, s->d_xb, dim);

    // dim3 swiglu_grid((hidden_dim + 255) / 256);
    // dim3 swiglu_block(256);
    // swiglu_kernel_cublas_v2<<<swiglu_grid, swiglu_block>>>(s->d_hb, s->d_hb, s->d_hb2, hidden_dim);
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Strict isolation mode: avoid cuBLAS in profiling path as well.
    dim3 swiglu_grid((hidden_dim + 255) / 256);
    dim3 swiglu_block(256);
    swiglu_kernel_cublas_v2<<<swiglu_grid, swiglu_block>>>(s->d_hb, s->d_hb, s->d_hb2, hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_matmul_2d_tiled(s->d_xb, s->d_hb, layer_w2, hidden_dim, dim);
    cuda_elementwise_add(s->d_x, s->d_xb, dim);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    stage8_ms = elapsed_time;
    printf("8. SwiGLU + W2 :               %.3f ms  \n", elapsed_time);
    total_time += elapsed_time;
    
    // 9. Final Residual
    // cudaEventRecord(start);
    // cuda_saxpy(s->d_x, s->d_xb, alpha, dim);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsed_time, start, stop);
    // printf("9. Final Residual:             %.3f ms \n", elapsed_time);
    // total_time += elapsed_time;
    
    float est_32_layer_ms = total_time * 32.0f;
    float est_tok_s = 1000.0f / est_32_layer_ms;
    printf("\nTOTAL LAYER TIME: %.3f ms\n", total_time);
    printf("ESTIMATED 32-LAYER TIME: %.1f ms = %.1f tok/s\n", est_32_layer_ms, est_tok_s);

    append_profile_metrics_csv(
        token,
        pos,
        dim,
        hidden_dim,
        p->n_heads,
        p->n_kv_heads,
        stage1_ms,
        stage2_ms,
        stage3_ms,
        stage4_ms,
        stage5_ms,
        stage6_ms,
        stage7_ms,
        stage8_ms,
        total_time,
        est_32_layer_ms,
        est_tok_s);
    printf("Profiling metrics appended to %s\n", g_profile_csv_path);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

GPUCapabilities detect_gpu_capabilities() {
    GPUCapabilities caps;
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    caps.major = prop.major;
    caps.minor = prop.minor;
    caps.multiProcessorCount = prop.multiProcessorCount;
    caps.sharedMemPerBlock = prop.sharedMemPerBlock;
    caps.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    strcpy(caps.name, prop.name);
    
    return caps;
}

// Detect GPU capabilities and choose optimal configuration
__host__ GPUConfig detect_gpu_config() {
    static GPUConfig config = {0};
    static bool initialized = false;
    
    if (!initialized) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        config.major = prop.major;
        config.minor = prop.minor;
        config.sm_count = prop.multiProcessorCount;
        config.shared_mem_per_block = prop.sharedMemPerBlock;
        config.max_threads_per_block = prop.maxThreadsPerBlock;
        
        // Detect specific GPU types
        config.is_a100 = (prop.major == 8 && prop.minor == 0 && prop.multiProcessorCount >= 100);
        config.is_rtx20_series = (prop.major == 7 && prop.minor == 5);
        
        printf("GPU Config: %s, SM=%d.%d, SMs=%d, SharedMem=%zuKB\n",
               prop.name, config.major, config.minor, config.sm_count, 
               config.shared_mem_per_block / 1024);
        
        initialized = true;
    }
    return config;
}

// Configure shared memory for all kernels based on GPU capabilities
void configure_shared_memory_limits() {
    GPUConfig config = detect_gpu_config();
    size_t max_shared_mem = 48 * 1024;  // Default 48KB
    
    // Determine maximum shared memory based on GPU
    if (config.is_a100) {
        max_shared_mem = 48 * 1024;  // A100: 163 KB
    } else if (config.major >= 9) {
        max_shared_mem = 227 * 1024;  // H100/newer: 227 KB
    } else if (config.major >= 8) {
        max_shared_mem = 99 * 1024;   // RTX 30xx/40xx: 99 KB
    } else if (config.major == 7) {
        max_shared_mem = (config.minor == 0) ? 96 * 1024 : 64 * 1024;  // Volta: 96KB, Turing: 64KB
    } else if (config.major >= 6) {
        max_shared_mem = 64 * 1024;   // Pascal: 64KB
    }
    
    // printf("Configuring shared memory limit: %zu KB for all kernels\n", max_shared_mem / 1024);
    
    // Set shared memory for all kernels that use dynamic shared memory
    cudaError_t err;
    
    // QKV kernels
    // err = cudaFuncSetAttribute(qkv_kernel_a100_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for qkv_kernel_a100_optimized\n");
    
    // err = cudaFuncSetAttribute(qkv_kernel_multiblock_a100, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for qkv_kernel_multiblock_a100\n");
    
    // err = cudaFuncSetAttribute(qkv_kernel_rtx2070_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for qkv_kernel_rtx2070_optimized\n");
    
    // err = cudaFuncSetAttribute(fused_qkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_qkv_kernel\n");
    
    // // FFN kernels
    // err = cudaFuncSetAttribute(fused_ffn_w1w3_chunked_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_ffn_w1w3_chunked_kernel\n");
    
    // err = cudaFuncSetAttribute(fused_ffn_w1w3_optimized_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_ffn_w1w3_optimized_kernel\n");
    
    // err = cudaFuncSetAttribute(fused_ffn_w1w3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_ffn_w1w3_kernel\n");
    
    // err = cudaFuncSetAttribute(improved_ffn_w1w3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for improved_ffn_w1w3_kernel\n");
    
    // err = cudaFuncSetAttribute(memory_optimized_ffn_w1w3_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for memory_optimized_ffn_w1w3_kernel\n");
    
    // err = cudaFuncSetAttribute(ffn_w1w3_a100_ultra_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for ffn_w1w3_a100_ultra_optimized\n");
    
    // // Output projection kernels
    // err = cudaFuncSetAttribute(output_projection_a100_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for output_projection_a100_optimized\n");
    
    // err = cudaFuncSetAttribute(output_projection_rtx2070_optimized, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for output_projection_rtx2070_optimized\n");
    
    // // Swiglu kernels
    // err = cudaFuncSetAttribute(fused_swiglu_w2_chunked_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_swiglu_w2_chunked_kernel\n");
    
    // err = cudaFuncSetAttribute(fused_swiglu_w2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_swiglu_w2_kernel\n");
    
    // // Other kernels
    // err = cudaFuncSetAttribute(fused_output_residual_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for fused_output_residual_kernel\n");
    
    // err = cudaFuncSetAttribute(multi_head_flashattention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for multi_head_flashattention_kernel\n");
    
    // err = cudaFuncSetAttribute(sgemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    // if (err != cudaSuccess) printf("Warning: Failed to set shared memory for sgemv_kernel\n");
    
    printf("Shared memory configuration completed!\n");
}

// Updated : For Large models, we need to adaptively choose the right kernel based on GPU capabilities
// Adaptive wrapper that chooses the right kernel
void cuda_rmsnorm_adaptive(
    float* d_output,
    float* d_input,
    float* d_weight,
    int size
) {
    static GPUCapabilities caps = detect_gpu_capabilities();
    static bool caps_initialized = false;
    
    if (!caps_initialized) {
        printf("Detected GPU: %s\n", caps.name);
        printf("Compute Capability: %d.%d\n", caps.major, caps.minor);
        printf("SMs: %d, Shared Memory: %zu KB\n", 
               caps.multiProcessorCount, caps.sharedMemPerBlock / 1024);
        caps_initialized = true;
    }
    
    // Choose optimal configuration based on GPU
    int block_size, grid_size;
    size_t shared_mem_size;
    
    if (caps.major >= 8) {
        // A100, A6000, RTX 3090, RTX 4090, etc.
        block_size = 1024;
        grid_size = 1;
        shared_mem_size = 128;
        
        // printf("Using A100-optimized kernel with %d threads per block\n", block_size);

        dim3 blockDim(block_size);
        dim3 gridDim(grid_size);
        
        rmsnorm_kernel_a100_optimized<<<gridDim, blockDim, shared_mem_size>>>(
            d_output, d_input, d_weight, size, 1e-5f
        );
    }
    else if (caps.major == 7 && caps.minor == 5) {
        // RTX 2070 Super, RTX 2080 Ti, etc.
        block_size = 256;  // Conservative for better occupancy
        grid_size = 1;
        shared_mem_size = 128;  // Well within 96KB limit
        
        dim3 blockDim(block_size);
        dim3 gridDim(grid_size);
        
        rmsnorm_kernel_rtx2070_optimized<<<gridDim, blockDim, shared_mem_size>>>(
            d_output, d_input, d_weight, size, 1e-5f
        );
    }
    else {
        // Older GPUs - use conservative approach
        block_size = 256;
        grid_size = (size + block_size - 1) / block_size;
        shared_mem_size = 32 * sizeof(float);
        
        dim3 blockDim(block_size);
        dim3 gridDim(grid_size);
        
        // Use your original kernel for older GPUs
        rmsnorm_kernel<<<gridDim, blockDim>>>(
            d_output, d_input, d_weight, size, 1e-5f
        );
    }
}


// RTX 2070 Super specific optimizations
void cuda_rmsnorm_rtx2070_optimized(
    float* d_output,
    float* d_input,
    float* d_weight,
    int size,
    int layer,
    int pos,
    int print_debug
) {
    // RTX 2070 Super: 40 SMs, 48KB shared memory, compute 7.5
    
    const int block_size = 256;  // Good balance for RTX 2070 Super
    const int shared_mem_size = 128;  // Conservative shared memory usage
    
    dim3 blockDim(block_size);
    dim3 gridDim(1);  // Single block handles small tensors efficiently
    
    rmsnorm_kernel_rtx2070_optimized<<<gridDim, blockDim, shared_mem_size>>>(
        d_output, d_input, d_weight, size, 1e-5f
    );
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

void cuda_matmul(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 64;  // started with 256, 64 does better
    int numBlocks = (d + blockSize - 1) / blockSize;
    matmul_kernel_simple<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Host functions for each version
void cuda_matmul_tiled(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 256;   //  256 < 128 < 64 < 32 --> Better
    int numBlocks = (d + blockSize - 1) / blockSize;
    matmul_kernel_tiled<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul_coarsened(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = 256;   // 256(<8>), 256(<4>) < 32(<4>)
    int numBlocks = ((d + COARSE_FACTOR - 1) / COARSE_FACTOR + blockSize - 1) / blockSize;
    matmul_kernel_coarsened<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_matmul_optimized(float* d_xout, float* d_x, float* d_w, int n, int d) {
    // 256(<8>) - 1.39, 256(<4>) - 2.5 t/s ,  32(<4>)  2.51 t/s , 256(<2>) - 4.46  , 32(<2>) - 7.54 t/s

    int blockSize = 32; 
    int numBlocks = ((d + COARSE_FACTOR - 1) / COARSE_FACTOR + blockSize - 1) / blockSize;
    matmul_kernel_optimized<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cuda_swiglu(float* d_hb, float* d_hb2, int size) {
    int blockSize = 320;    // started with 256, [64,1024 - (worse than 256)] 
    int numBlocks = (size + blockSize - 1) / blockSize;
    swiglu_kernel<<<numBlocks, blockSize>>>(d_hb, d_hb2, size);
    CUDA_CHECK(cudaGetLastError());
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


    if (g_enable_profile && !g_profile_triggered && pos == g_profile_pos) {
        profile_7b_bottlenecks(transformer, token, pos);
        g_profile_triggered = true;
    }

#if USE_CUDA
    if (!g_cuda_available) {
        return forward_fixed(transformer, token, pos);  // Fall back to CPU
    }

    // Copy token embedding to GPU
    float* token_embedding = w->token_embedding_table + token * dim;
    CUDA_CHECK(cudaMemcpy(s->d_x, token_embedding, dim * sizeof(float), cudaMemcpyHostToDevice));

    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        
        float* layer_rms_att_weight = w->d_rms_att_weight + l * dim;
        // cuda_rmsnorm(s->d_xb, s->d_x, layer_rms_att_weight, dim);

        cuda_rmsnorm_adaptive(
            s->d_xb,                  // Output tensor
            s->d_x,                   // Input tensor
            layer_rms_att_weight,     // RMSNorm weights
            dim                       // Tensor size (dim)
        );

       
        // -------------------------------------------------------------------------------
        
        // 2. Profile QKV Projections
        // QKV matmuls using cuBLAS

        // err = cudaEventRecord(start);

        float* layer_wq = w->d_wq + l * dim * dim;
        float* layer_wk = w->d_wk + l * dim * kv_dim;
        float* layer_wv = w->d_wv + l * dim * kv_dim;

        // cuda_fused_qkv_projection(
        //     s->d_xb,        // Input vector [dim]
        //     layer_wq,       // Query weights [dim x dim]
        //     layer_wk,       // Key weights [dim x kv_dim] 
        //     layer_wv,       // Value weights [dim x kv_dim]
        //     s->d_q,         // Query output [dim]
        //     s->d_k,         // Key output [kv_dim]
        //     s->d_v,         // Value output [kv_dim]
        //     dim,            // Input dimension
        //     kv_dim,         // KV dimension
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // cuda_qkv_adaptive_optimized(
        //     s->d_xb,        // Input vector [dim]
        //     layer_wq,       // Query weights [dim x dim]
        //     layer_wk,       // Key weights [dim x kv_dim] 
        //     layer_wv,       // Value weights [dim x kv_dim]
        //     s->d_q,         // Query output [dim]
        //     s->d_k,         // Key output [kv_dim]
        //     s->d_v,         // Value output [kv_dim]
        //     dim,            // Input dimension
        //     kv_dim,         // KV dimension
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // cuda_matmul(s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
        // cuda_matmul(s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
        // cuda_matmul(s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);

        // Before
        // cuda_matmul_2d_tiled(s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
        // cuda_matmul_2d_tiled(s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
        // cuda_matmul_2d_tiled(s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);

        // cuda_matmul_2d_vectorized (s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
        // cuda_matmul_2d_vectorized (s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
        // cuda_matmul_2d_vectorized (s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);

        // Strict isolation mode: use custom CUDA kernels for Q/K/V projections.
        cuda_matmul_2d_tiled(s->d_q, s->d_xb, layer_wq, dim, dim);
        cuda_matmul_2d_tiled(s->d_k, s->d_xb, layer_wk, dim, kv_dim);
        cuda_matmul_2d_tiled(s->d_v, s->d_xb, layer_wv, dim, kv_dim);


        // -------------------------------------------------------------------------------
       
        // Apply RoPE on GPU
        // 3. Profile RoPE

        // cudaEventRecord(start);

        int total_elements = 1 * 1 * p->n_heads * head_size;
        dim3 blockDim(256);
        dim3 gridDim((total_elements / 2 + blockDim.x - 1) / blockDim.x);  // Process pairs
        
        rope_kernel<<<gridDim, blockDim>>>(s->d_q, s->d_k, 1, 1, p->n_heads, head_size, pos);
        CUDA_CHECK(cudaDeviceSynchronize());

        // -------------------------------------------------------------------------------
        
        // 4. Profile KV Cache Update
        // Update KV cache on GPU
        
        // cudaEventRecord(start);
        
        int loff = l * p->seq_len * kv_dim;
        float* d_key_cache_row = s->d_key_cache + loff + pos * kv_dim;
        float* d_value_cache_row = s->d_value_cache + loff + pos * kv_dim;
        
        CUDA_CHECK(cudaMemcpy(d_key_cache_row, s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_value_cache_row, s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        

        // -------------------------------------------------------------------------------

        // 5. Profile FlashAttention


        // cudaEventRecord(start);

        float* d_key_cache_layer = s->d_key_cache + loff;      // [seq_len * n_kv_heads * head_size]
        float* d_value_cache_layer = s->d_value_cache + loff;  // [seq_len * n_kv_heads * head_size]

        // Clear output buffer
        CUDA_CHECK(cudaMemset(s->d_xb, 0, dim * sizeof(float)));

        // Calculate block sizes
        const int M = 48 * 1024;  // 48KB for RTX 2070 Super
        const int Bc = min(M / (4 * head_size), pos + 1);  // Don't exceed sequence length
        const int actual_Bc = min(Bc, 32);  // Reasonable limit

        // Get A100-specific memory size
        // const int M = get_optimal_block_size_a100(head_size, 0);
        
        // // Calculate optimal block sizes for A100
        // int optimal_Bc = M / (4 * head_size);
        
        // // A100 can handle much larger blocks
        // const int Bc = min(optimal_Bc, pos + 1);
        // const int actual_Bc = min(Bc, 128);  // Increase from 32 to 128


        // Calculate shared memory size
        size_t smem_size = sizeof(float) * (
            actual_Bc * head_size +    // Kj
            actual_Bc * head_size +    // Vj
            head_size +                // Qi
            actual_Bc +                // Sij
            actual_Bc +                // Pij
            head_size +                // Oi_local
            6                          // Statistics
        );

        // Launch multi-head kernel (one block per head)
        dim3 grid_fa(p->n_heads);
        dim3 block_fa(256);

        multi_head_flashattention_kernel<<<grid_fa, block_fa, smem_size>>>(
            s->d_q,                    // All queries
            d_key_cache_layer,         // Key cache
            d_value_cache_layer,       // Value cache
            s->d_xb,                   // All outputs
            pos + 1,                   // Sequence length
            head_size,                 // Head dimension
            p->n_heads,                // Number of heads
            p->n_kv_heads,             // Number of KV heads
            p->n_kv_heads * head_size, // KV stride
            scale,                      // Scale
            actual_Bc                  // Block size
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // -------------------------------------------------------------------------------

        // 6. Profile Output Projection + Residual
        // Fuse Output Projection + Residual (Expected: 8-12% gain)  ----- 
        // Combining operations : Output projection: xb2 = xb * Wo^T  and Residual connection: x = x + xb2


        // cudaEventRecord(start);        

        float* layer_wo = w->d_wo + l * dim * dim;
        
        // cuda_fused_output_residual(
        //     s->d_xb,     // Attention output
        //     layer_wo,    // Output projection weights
        //     s->d_x,      // Input/output (in-place residual)
        //     dim, alpha
        // );

        // cuda_output_projection_adaptive_optimized(
        //     s->d_xb,     // Attention output
        //     layer_wo,    // Output projection weights
        //     s->d_x,      // Input/output (in-place residual)
        //     dim, alpha
        // );

        cuda_matmul_2d_tiled(s->d_xb2, s->d_xb, layer_wo, dim, dim);
        cuda_elementwise_add(s->d_x, s->d_xb2, dim);

        // -------------------------------------------------------------------------------

        // 7. Profile FFN RMSNorm
        // FFN rmsnorm

        // cudaEventRecord(start);

        float* layer_rms_ffn_weight = w->d_rms_ffn_weight + l * dim;
        // cuda_rmsnorm(s->d_xb, s->d_x, layer_rms_ffn_weight, dim);
        cuda_rmsnorm_adaptive(
            s->d_xb,                  // Output tensor
            s->d_x,                   // Input tensor
            layer_rms_ffn_weight,     // RMSNorm weights
            dim                       // Tensor size (dim)
        );


        // -------------------------------------------------------------------------------
        
        // 8. Profile FFN W1 and W3 Projections
        // FFN: w1 and w3 projections

        // cudaEventRecord(start); 

        float* layer_w1 = w->d_w1 + l * dim * hidden_dim;
        float* layer_w3 = w->d_w3 + l * dim * hidden_dim;

        // Fused W1 and W3 projections
        // cuda_fused_ffn_w1w3_optimized(
        //     s->d_xb,        // Input vector [dim]
        //     layer_w1,       // Gate weights W1 [dim x hidden_dim]
        //     layer_w3,       // Up weights W3 [dim x hidden_dim]
        //     s->d_hb,        // Gate output [hidden_dim]
        //     s->d_hb2,       // Up output [hidden_dim]
        //     dim,            // Input dimension (4096)
        //     hidden_dim,     // Hidden dimension (11008)
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // cuda_improved_ffn_w1w3(
        //     s->d_xb,        // Input vector [dim]
        //     layer_w1,       // Gate weights W1 [dim x hidden_dim]
        //     layer_w3,       // Up weights W3 [dim x hidden_dim]
        //     s->d_hb,        // Gate output [hidden_dim]
        //     s->d_hb2,       // Up output [hidden_dim]
        //     dim,            // Input dimension (4096)
        //     hidden_dim,     // Hidden dimension (11008)
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // Before : 
        // cuda_memory_optimized_ffn_w1w3(
        //     s->d_xb,        // Input vector [dim]
        //     layer_w1,       // Gate weights W1 [dim x hidden_dim]
        //     layer_w3,       // Up weights W3 [dim x hidden_dim]
        //     s->d_hb,        // Gate output [hidden_dim]
        //     s->d_hb2,       // Up output [hidden_dim]
        //     dim,            // Input dimension (4096)
        //     hidden_dim,     // Hidden dimension (11008)
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // Strict isolation mode: use custom CUDA kernels for FFN W1/W3.
        cuda_matmul_2d_tiled(s->d_hb, s->d_xb, layer_w1, dim, hidden_dim);
        cuda_matmul_2d_tiled(s->d_hb2, s->d_xb, layer_w3, dim, hidden_dim);

        // cuda_matmul(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
        // cuda_matmul(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);

        // cuda_matmul_2d_tiled(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
        // cuda_matmul_2d_tiled(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);

        // -------------------------------------------------------------------------------


        // 9 and 10: Fused SwiGLU and FFN W2 projection

        dim3 swiglu_grid((hidden_dim + 255) / 256);
        dim3 swiglu_block(256);
        // float* layer_w2 = w->d_w2 + l * hidden_dim * dim;

        // cuda_fused_swiglu_w2_projection(
        //     s->d_hb,        // Gate input [hidden_dim]
        //     s->d_hb2,       // Up input [hidden_dim]
        //     layer_w2,       // W2 weights [hidden_dim x dim]
        //     s->d_xb,        // Output [dim]
        //     hidden_dim,     // Hidden dimension (11008)
        //     dim,            // Output dimension (4096)
        //     alpha,          // Scaling factor (1.0f)
        //     beta            // Beta factor (0.0f)
        // );

        // cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);
        // cuda_matmul(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
        // cuda_elementwise_add(s->d_x, s->d_xb, dim);

        // Before:
        // cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);
        // cuda_matmul_2d_tiled(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
        // cuda_elementwise_add(s->d_x, s->d_xb, dim);

        // ------ Best So far ------
        swiglu_kernel_cublas_v2<<<swiglu_grid, swiglu_block>>>(s->d_hb, s->d_hb, s->d_hb2, hidden_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Final FFN projection + residual using custom CUDA kernels.
        float* layer_w2 = w->d_w2 + l * hidden_dim * dim;
        cuda_matmul_2d_tiled(s->d_xb, s->d_hb, layer_w2, hidden_dim, dim);
        cuda_elementwise_add(s->d_x, s->d_xb, dim);
        //---------------------------
        


    }

    // err = cudaEventDestroy(start);
    // err = cudaEventDestroy(stop);
    // printf("=== End Layer %d Profiling ===\n\n");

    // Final rmsnorm
    // cuda_rmsnorm(s->d_x, s->d_x, w->d_rms_final_weight, dim);
    cuda_rmsnorm_adaptive(
            s->d_x,                  // Output tensor
            s->d_x,                   // Input tensor
            w->d_rms_final_weight,     // RMSNorm weights
            dim                       // Tensor size (dim)
        );

    // Classifier: logits = x * Wcls^T (strict isolation: custom CUDA kernel only)
    cuda_matmul_2d_tiled(s->d_logits, s->d_x, w->d_wcls, dim, p->vocab_size);

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
    printf("Strict isolation: cuBLAS disabled in flash forward path\n");
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
        printf("Achieved tok/s: %.2f\n", (pos-1) / (double)(end-start) * 1000);  // Also print to stdout for console visibility
    
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
        printf("Achieved tok/s: %.2f\n", (pos-1) / (double)(end-start) * 1000);  // Also print to stdout for console visibility
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
    fprintf(stderr, "  -P <int>    enable profiling and trigger at token position (e.g., 10)\n");
    fprintf(stderr, "  -R <string> profiling CSV output path (default: flash_profile_metrics.csv)\n");
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
        
        // Configure shared memory limits based on GPU capability
        if (prop.major >= 8) {
            // A100/V100/RTX30xx/RTX40xx: Enable large shared memory
            printf("GPU supports large shared memory (>= Ampere architecture)\n");
            if (prop.major == 8 && prop.minor == 0 && prop.multiProcessorCount >= 100) {
                printf("Detected A100 GPU - 163 KB shared memory available\n");
            } else if (prop.major >= 8) {
                printf("Detected RTX 30xx/40xx series - 99 KB shared memory available\n");
            }
        } else if (prop.major == 7) {
            // RTX 20xx/V100: 96 KB for Volta, 64 KB for Turing
            size_t max_shared_mem = (prop.minor == 0) ? 96 : 64;
            printf("Detected RTX 20xx/V100 series - %zu KB shared memory available\n", max_shared_mem);
        }

        printf("Initializing CUDA libraries...\n");
        // cublasCreate(&cublas_handle);
        // cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH); // Enable Tensor Cores
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH); // Enable Tensor Cores
        printf("CUDA initialization complete!\n");
        
        // Configure shared memory for all kernels
        configure_shared_memory_limits();
        
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
        else if (argv[i][1] == 'P') {
            g_enable_profile = true;
            g_profile_pos = atoi(argv[i + 1]);
            if (g_profile_pos < 0) { g_profile_pos = 0; }
        }
        else if (argv[i][1] == 'R') {
            g_enable_profile = true;
            strncpy(g_profile_csv_path, argv[i + 1], sizeof(g_profile_csv_path) - 1);
            g_profile_csv_path[sizeof(g_profile_csv_path) - 1] = '\0';
        }
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
    if (g_enable_profile) {
        printf("  Profiling: enabled (trigger pos=%d, csv=%s)\n", g_profile_pos, g_profile_csv_path);
    }
    
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

    // Cleanup
    printf("\nCleaning up...\n");
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    // Cleanup CUDA
    if (cuda_available) {
        cublasDestroy(cublas_handle);
    }
    
    printf("Done.\n");
    
    // Close logging system
    close_logging();
    
    return 0;
}
