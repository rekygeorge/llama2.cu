/* CUDA-Parallelized Llama-2 with Fixed Kernels */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if defined _WIN32
    #include <io.h> 
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
    // #include <fcntl.h>
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define USE_CUDA 1  // Re-enable CUDA with fixed kernels
#define TILE_SIZE 32

// Forward declarations for structs
struct GPUCapabilities;

// Forward declarations for functions (will be properly declared after struct definitions)
GPUCapabilities detect_gpu_capabilities();

// Global variables
static bool g_cuda_available = false;
static FILE* g_log_file = NULL;
static bool g_enable_profile = false;
static bool g_profile_triggered = false;
static int g_profile_pos = 10;
static char g_profile_csv_path[512] = "tiled_profile_metrics.csv";
#ifdef DUMP_LOGITS
static char g_logit_bin_path[512] = "";  /* -L flag: full path for raw float32 logit dump */
static char g_token_ids_path[512] = "";  /* -T flag: full path for generated token IDs JSON */
#endif

// Logging utilities
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
    if (!g_enable_profile || !g_profile_triggered) return;
    
    FILE* csv_file = fopen(g_profile_csv_path, "a");
    if (!csv_file) {
        fprintf(stderr, "Error: could not open %s for writing\n", g_profile_csv_path);
        return;
    }
    
    // Check if file is empty to write header
    fseek(csv_file, 0, SEEK_END);
    long file_size = ftell(csv_file);
    
    if (file_size == 0) {
        fprintf(csv_file,
                "timestamp_unix,token,pos,dim,hidden_dim,n_heads,n_kv_heads,"
                "rms_att_ms,qkv_ms,rope_kvcache_ms,cpu_attn_ms,h2d_out_proj_res_ms,"
                "rms_ffn_ms,ffn_w1w3_ms,swiglu_w2_ms,total_layer_ms,est_32_layer_ms,est_tok_s\n");
    }

    fprintf(csv_file,
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
    fclose(csv_file);
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

// Transformer structures 

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

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
    
    float* d_token_embedding_table;
    float* d_rms_att_weight;
    float* d_rms_ffn_weight;
    float* d_wq;
    float* d_wk;
    float* d_wv;
    float* d_wo;
    float* d_w1;
    float* d_w2;
    float* d_w3;
    float* d_rms_final_weight;
    float* d_wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float* key_cache, *value_cache;
    float* k_original, *v_original;
    
    float *d_x, *d_xb, *d_xb2, *d_hb, *d_hb2, *d_q, *d_k, *d_v, *d_att, *d_logits;
    float* d_key_cache, *d_value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// ----------------------------------------------------------------------------
// Function declarations

float* forward(Transformer* transformer, int token, int pos);

// ----------------------------------------------------------------------------
// FIXED CUDA Kernels

// Shared-memory tiled matrix-vector multiplication kernel.
// Each block caches a tile of x to reduce global memory traffic.
__global__ void matmul_kernel_tiled(float* xout, float* x, float* w, int n, int d) {
    __shared__ float shared_x[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i >= d) {
        return;
    }

    float val = 0.0f;
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        if (tid < TILE_SIZE && tile_start + tid < n) {
            shared_x[tid] = x[tile_start + tid];
        } else if (tid < TILE_SIZE) {
            shared_x[tid] = 0.0f;
        }
        __syncthreads();

        int tile_end = n - tile_start;
        if (tile_end > TILE_SIZE) {
            tile_end = TILE_SIZE;
        }
        for (int j = 0; j < tile_end; j++) {
            val += w[i * n + tile_start + j] * shared_x[j];
        }
        __syncthreads();
    }

    xout[i] = val;
}

// Fixed RMSNorm kernel - compute norm factor on CPU, apply on GPU
__global__ void rmsnorm_apply_kernel(float* o, float* x, float* weight, float norm_factor, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        o[i] = weight[i] * (norm_factor * x[i]);
    }
}

// Element-wise kernels (these were working fine)
__global__ void elementwise_add_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

__global__ void swiglu_kernel(float* hb, float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Hybrid CPU/GPU wrapper functions

void cuda_matmul(float* d_xout, float* d_x, float* d_w, int n, int d) {
    int blockSize = TILE_SIZE;
    int numBlocks = (d + blockSize - 1) / blockSize;
    matmul_kernel_tiled<<<numBlocks, blockSize>>>(d_xout, d_x, d_w, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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


// Hybrid RMSNorm: compute norm factor on CPU, apply normalization on GPU
void cuda_rmsnorm_hybrid(float* d_o, float* d_x, float* d_weight, float* h_x, int size, int layer, int pos, int print_debug) {
    
    
    static GPUCapabilities caps = detect_gpu_capabilities();
    static bool caps_initialized = false;
    
    if (!caps_initialized) {
        printf("Detected GPU: %s\n", caps.name);
        printf("Compute Capability: %d.%d\n", caps.major, caps.minor);
        printf("SMs: %d, Shared Memory: %zu KB\n", 
               caps.multiProcessorCount, caps.sharedMemPerBlock / 1024);
        caps_initialized = true;
    }

    // Step 1: Copy data to host for norm computation
    CUDA_CHECK(cudaMemcpy(h_x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 2: Compute norm factor on CPU (more reliable)
    double ss = 0.0;
    for (int j = 0; j < size; j++) {
        double val = (double)h_x[j];
        ss += val * val;
    }
    ss /= size;
    ss += 1e-5;
    float norm_factor = (float)(1.0 / sqrt(ss));
    
    // if (pos < 10 && layer == 0 && print_debug) {
    //     printf("print_debug rms_scale: %f\n", norm_factor);
    // }

    // Step 3: Apply normalization on GPU
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    rmsnorm_apply_kernel<<<numBlocks, blockSize>>>(d_o, d_x, d_weight, norm_factor, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_elementwise_add(float* d_a, float* d_b, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    elementwise_add_kernel<<<numBlocks, blockSize>>>(d_a, d_b, size);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_swiglu(float* d_hb, float* d_hb2, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    swiglu_kernel<<<numBlocks, blockSize>>>(d_hb, d_hb2, size);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// CPU fallback functions (same reliable versions from debug)

void rmsnorm(float* o, float* x, float* weight, int size) {
    double ss = 0.0;
    for (int j = 0; j < size; j++) {
        double val = (double)x[j];
        ss += val * val;
    }
    ss /= size;
    ss += 1e-5;
    float norm_factor = (float)(1.0 / sqrt(ss));
    
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (norm_factor * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    
    float sum_f = (float)sum;
    for (int i = 0; i < size; i++) {
        x[i] /= sum_f;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        double val = 0.0;
        for (int j = 0; j < n; j++) {
            val += (double)w[i * n + j] * (double)x[j];
        }
        xout[i] = (float)val;
    }
}

// ----------------------------------------------------------------------------------
// Profiling utilities

// Profile function for the hybrid CPU-GPU forward function
void profile_7b_bottlenecks(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    printf("\n=== MODEL BOTTLENECK ANALYSIS (Layer 0, pos=%d) ===\n", pos);
    
    // Copy token embedding
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(s->x, content_row, dim * sizeof(float));
    CUDA_CHECK(cudaMemcpy(s->d_x, s->x, dim * sizeof(float), cudaMemcpyHostToDevice));
    
    unsigned long long l = 0; // Profile just first layer
    float total_time = 0.0f;
    
    // 1. GPU Attention RMSNorm
    cudaEventRecord(start);
    cuda_rmsnorm_hybrid(s->d_xb, s->d_x, w->d_rms_att_weight + l*dim, s->xb, dim, l, pos, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("1. RMSNorm (attention):        %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 2. GPU QKV Matrix Multiplications
    cudaEventRecord(start);
    cuda_matmul(s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
    cuda_matmul(s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
    cuda_matmul(s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("2. QKV Projections:            %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 3. GPU to CPU Transfer for Attention
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(s->q, s->d_q, dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s->k_original, s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s->v_original, s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("3. QKV->CPU Transfer:          %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 4. CPU Cache Management
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int loff = l * p->seq_len * kv_dim;
    float* k_cache_ptr = s->key_cache + loff + pos * kv_dim;
    float* v_cache_ptr = s->value_cache + loff + pos * kv_dim;
    memcpy(k_cache_ptr, s->k_original, kv_dim * sizeof(float));
    memcpy(v_cache_ptr, s->v_original, kv_dim * sizeof(float));
    s->k = k_cache_ptr;
    s->v = v_cache_ptr;
    auto cpu_end = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("4. KV Cache Write (CPU):       %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 5. CPU RoPE Encoding
    cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
    cpu_end = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("5. RoPE Encoding (CPU):        %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 6. CPU Multi-Head Attention (MAJOR BOTTLENECK)
    cpu_start = std::chrono::high_resolution_clock::now();
    for (int h = 0; h < p->n_heads; h++) {
        float* q = s->q + h * head_size;
        float* att = s->att + h * p->seq_len;
        
        // Attention scores
        for (int t = 0; t <= pos; t++) {
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            att[t] = score;
        }
        
        // Softmax
        softmax(att, pos + 1);
        
        // Apply attention weights
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[t];
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
    cpu_end = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    printf("6. CPU Multi-Head Attention:   %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 7. CPU to GPU Transfer for FFN
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(s->d_xb, s->xb, dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("7. Attn->GPU Transfer:         %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 8. GPU Output Projection + Residual
    cudaEventRecord(start);
    cuda_matmul(s->d_xb2, s->d_xb, w->d_wo + l*dim*dim, dim, dim);
    cuda_elementwise_add(s->d_x, s->d_xb2, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("8. Output Proj + Residual:     %.3f ms \n", elapsed_time);
    total_time += elapsed_time;
    
    // 9-12. Additional tiled-only work is intentionally not printed so the
    // visible profiling output matches the flash format.
    cudaEventRecord(start);
    cuda_rmsnorm_hybrid(s->d_xb, s->d_x, w->d_rms_ffn_weight + l*dim, s->xb, dim, l, pos, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    
    cudaEventRecord(start);
    cuda_matmul(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
    cuda_matmul(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    
    cudaEventRecord(start);
    cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    
    cudaEventRecord(start);
    cuda_matmul(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
    cuda_elementwise_add(s->d_x, s->d_xb, dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_time += elapsed_time;
    
    const float total_layer_ms = total_time;
    const float est_32_layer_ms = total_layer_ms * 32.0f;
    const float est_tok_s = est_32_layer_ms > 0.0f ? (1000.0f / est_32_layer_ms) : 0.0f;
    printf("\nTOTAL LAYER TIME: %.3f ms\n", total_layer_ms);
    printf("ESTIMATED 32-LAYER TIME: %.1f ms = %.1f tok/s\n", est_32_layer_ms, est_tok_s);
    
    // printf("\n=== BOTTLENECK ANALYSIS ===\n");
    // printf("Major Issues in This Hybrid Approach:\n");
    // printf("1. CPU Multi-Head Attention: Likely 50-80%% of total time\n");
    // printf("   - Single-threaded CPU vs massively parallel GPU\n");
    // printf("   - O(seq_len × n_heads × head_size) complexity\n");
    // printf("   - No vectorization or SIMD optimization\n\n");
    
    // printf("2. GPU↔CPU Memory Transfers: Significant overhead\n");
    // printf("   - Transfer QKV: ~49KB per layer\n");
    // printf("   - Transfer attention results: ~16KB per layer\n");
    // printf("   - PCIe bandwidth: ~25GB/s (much slower than GPU memory)\n\n");
    
    // printf("3. Synchronization Overhead:\n");
    // printf("   - cudaMemcpy calls force GPU-CPU synchronization\n");
    // printf("   - Prevents overlapping compute with memory transfers\n\n");
    
    // printf("RECOMMENDATIONS:\n");
    // printf("1. Move attention computation to GPU (FlashAttention)\n");
    // printf("2. Eliminate GPU↔CPU transfers within layers\n");
    // printf("3. Use CUDA streams for async operations\n");
    // printf("4. Consider full GPU pipeline for 5-10x improvement\n");
    // printf("========================================\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Alternative: Quick timing version without detailed breakdown
void profile_hybrid_quick(Transformer* transformer, int token, int pos) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;
    
    printf("\n=== QUICK HYBRID PROFILE (pos=%d) ===\n", pos);
    
    cudaEventRecord(start);
    float* logits = forward(transformer, token, pos);  // Your existing forward function
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("TOTAL FORWARD PASS: %.3f ms\n", elapsed_time);
    printf("ESTIMATED THROUGHPUT: %.1f tok/s\n", 1000.0f / elapsed_time);
    printf("===================================\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Memory transfer analysis
void analyze_gpu_cpu_transfers(Config* p) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    
    printf("\n=== GPU↔CPU TRANSFER ANALYSIS ===\n");
    
    // Per layer transfers
    size_t qkv_transfer = (dim + 2 * kv_dim) * sizeof(float);  // Q, K, V to CPU
    size_t attention_transfer = dim * sizeof(float);           // Attention result to GPU
    size_t per_layer_transfer = qkv_transfer + attention_transfer;
    
    printf("Per Layer Transfers:\n");
    printf("  QKV (GPU→CPU): %zu bytes (%.1f KB)\n", qkv_transfer, qkv_transfer / 1024.0f);
    printf("  Attention (CPU→GPU): %zu bytes (%.1f KB)\n", attention_transfer, attention_transfer / 1024.0f);
    printf("  Total per layer: %zu bytes (%.1f KB)\n", per_layer_transfer, per_layer_transfer / 1024.0f);
    
    // Total for all layers
    size_t total_transfer = per_layer_transfer * p->n_layers;
    printf("\nTotal for %d layers: %zu bytes (%.1f MB)\n", 
           p->n_layers, total_transfer, total_transfer / (1024.0f * 1024.0f));
    
    // Transfer time estimates
    float pcie_bandwidth = 25.0f;  // GB/s (typical PCIe 4.0 x16)
    float transfer_time = (total_transfer / (1024.0f * 1024.0f * 1024.0f)) / pcie_bandwidth * 1000.0f;
    
    printf("Estimated transfer time: %.1f ms (at %.0f GB/s PCIe)\n", 
           transfer_time, pcie_bandwidth);
    printf("Transfer overhead: Significant bottleneck!\n");
    printf("=====================================\n\n");
}

// ----------------------------------------------------------------------------
// Memory management with temp buffer for hybrid operations

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    // Initialize pointers
    s->x = s->xb = s->xb2 = s->hb = s->hb2 = s->q = s->k = s->v = NULL;
    s->key_cache = s->value_cache = s->att = s->logits = NULL;
    s->k_original = s->v_original = NULL;
    s->d_x = s->d_xb = s->d_xb2 = s->d_hb = s->d_hb2 = NULL;
    s->d_q = s->d_k = s->d_v = s->d_att = s->d_logits = NULL;
    s->d_key_cache = s->d_value_cache = NULL;
    
    // Allocate host memory
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
    // Allocate device memory
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
#endif
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
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
    // ptr += p->vocab_size * p->dim; //
    size_t token_embed_size = p->vocab_size * p->dim;
    printf("1. Token Embedding Table:\n");
    printf("   - Dimensions: [%d, %d] (vocab_size × dim)\n", p->vocab_size, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", token_embed_size, token_embed_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Maps vocabulary tokens to dense vector embeddings\n");
    ptr += token_embed_size;

    // 2. ATTENTION RMSNORM WEIGHTS  
    // Dimensions: [n_layers, dim] - RMSNorm scaling parameters for attention layers
    // Memory: n_layers * dim * sizeof(float) bytes
    w->rms_att_weight = ptr;       //
    // ptr += n_layers * p->dim;      //
    size_t rms_att_size = p->n_layers * p->dim;
    printf("\n2. Attention RMSNorm Weights:\n");
    printf("   - Dimensions: [%d, %d] (n_layers × dim)\n", p->n_layers, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", rms_att_size, rms_att_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Scaling parameters for RMSNorm before self-attention\n");
    ptr += rms_att_size;

    // 3. QUERY PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, dim] - Projects input to query vectors for all attention heads
    // Memory: n_layers * dim * dim * sizeof(float) bytes
    w->wq = ptr;                   //
    // ptr += n_layers * p->dim * (p->n_heads * head_size);   // Corrected in Assignment_05
    size_t wq_size = p->n_layers * p->dim * (p->n_heads * head_size);
    printf("\n3. Query Projection Weights (Wq):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × dim)\n", p->n_layers, p->dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", wq_size, wq_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to query vectors for multi-head attention\n");
    ptr += wq_size;

    // 4. KEY PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, kv_head_size] - Projects input to key vectors
    // For multi-query attention, kv_head_size may be smaller than dim
    w->wk = ptr;                   //
    // ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // Corrected in Assignment_05
    size_t wk_size = n_layers * p->dim * (p->n_kv_heads * head_size);
    printf("\n4. Key Projection Weights (Wk):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim ×  head_size)\n", p->n_layers, p->dim, head_size);
    printf("   - Memory: %zu floats = %.2f MB\n", wk_size, wk_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to key vectors for attention computation\n");
    ptr += wk_size;

    // 5. VALUE PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, kv_head_size] - Projects input to value vectors
    // Symmetric to key projection for multi-query attention
    w->wv = ptr;                   //
    // ptr += n_layers * p->dim * (p->n_kv_heads * head_size); // Corrected in Assignment_05
    size_t wv_size = n_layers * p->dim * (p->n_kv_heads * head_size);
    printf("\n5. Value Projection Weights (Wv):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × kv_head_size)\n", p->n_layers, p->dim, head_size);
    printf("   - Memory: %zu floats = %.2f MB\n", wv_size, wv_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects input vectors to value vectors for attention computation\n");
    ptr += wv_size;

    // 6. OUTPUT PROJECTION WEIGHTS
    // Dimensions: [n_layers, dim, dim] - Projects concatenated attention heads back to model dimension
    // Memory: n_layers * dim * dim * sizeof(float) bytes
    w->wo = ptr;                   //
    // ptr += n_layers * (p->n_heads * head_size) * p->dim;    // Corrected in Assignment_05
    size_t wo_size = p->n_layers * (p->n_heads * head_size) * p->dim;
    printf("\n6. Output Projection Weights (Wo):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × dim)\n", p->n_layers, p->dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", wo_size, wo_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects concatenated attention head outputs back to model dimension\n");
    ptr += wo_size;

    // 7. FEEDFORWARD RMSNORM WEIGHTS
    // Dimensions: [n_layers, dim] - RMSNorm scaling parameters for feedforward layers
    // Memory: n_layers * dim * sizeof(float) bytes
    w->rms_ffn_weight = ptr;       //
    // ptr += n_layers * p->dim;      //
    size_t rms_ffn_size = n_layers * p->dim;
    printf("\n7. Feedforward RMSNorm Weights:\n");
    printf("   - Dimensions: [%d, %d] (n_layers × dim)\n", p->n_layers, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", rms_ffn_size, rms_ffn_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Scaling parameters for RMSNorm before feedforward network\n");
    ptr += rms_ffn_size;

    // 8. FEEDFORWARD GATE PROJECTION WEIGHTS (W1)
    // Dimensions: [n_layers, dim, hidden_dim] - First linear layer in SwiGLU feedforward
    // Memory: n_layers * dim * hidden_dim * sizeof(float) bytes
    w->w1 = ptr;                   //   
    // ptr += n_layers * p->dim * p->hidden_dim;     //
    size_t w1_size = n_layers * p->dim * p->hidden_dim;
    printf("\n8. Feedforward Gate Projection Weights (W1):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × dim × hidden_dim)\n", p->n_layers, p->dim, p->hidden_dim);
    printf("   - Memory: %zu floats = %.2f MB\n", w1_size, w1_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Gate projection in SwiGLU activation (part of feedforward network)\n");
    ptr += w1_size;

    // 9. FEEDFORWARD DOWN PROJECTION WEIGHTS (W2)
    // Dimensions: [n_layers, hidden_dim, dim] - Final linear layer in feedforward network
    // Memory: n_layers * hidden_dim * dim * sizeof(float) bytes
    w->w2 = ptr;                   //   
    // ptr += n_layers * p->hidden_dim * p->dim;     //
    size_t w2_size = n_layers * p->hidden_dim * p->dim;
    printf("\n9. Feedforward Down Projection Weights (W2):\n");
    printf("   - Dimensions: [%d, %d, %d] (n_layers × hidden_dim × dim)\n", p->n_layers, p->hidden_dim, p->dim);
    printf("   - Memory: %zu floats = %.2f MB\n", w2_size, w2_size * sizeof(float) / (1024.0f * 1024.0f));
    printf("   - Purpose: Projects from hidden dimension back to model dimension in feedforward\n");
    ptr += w2_size;

    // 10. FEEDFORWARD UP PROJECTION WEIGHTS (W3)
    // Dimensions: [n_layers, dim, hidden_dim] - Second linear layer in SwiGLU feedforward
    // Memory: n_layers * dim * hidden_dim * sizeof(float) bytes
    w->w3 = ptr;                   // 
    // ptr += n_layers * p->dim * p->hidden_dim;    //
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
    ptr += p->dim;                 

    // Skip frequency caches (used for RoPE positional encoding, pre-computed)
    size_t freq_cache_size = p->seq_len * head_size / 2;
    printf("\n12. Skipping RoPE Frequency Caches:\n");
    printf("    - Cos cache size: %zu floats\n", freq_cache_size);
    printf("    - Sin cache size: %zu floats\n", freq_cache_size);
    printf("    - Purpose: Pre-computed cosine/sine values for Rotary Position Embedding\n");
    ptr += freq_cache_size; // cos frequencies
    ptr += freq_cache_size; // sin frequencies
    // ptr += p->seq_len * head_size / 2;  //
    // ptr += p->seq_len * head_size / 2;  //


    // w->wcls = shared_weights ? w->token_embedding_table : ptr;
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
    
    // Initialize device pointers
    w->d_token_embedding_table = NULL;
    w->d_rms_att_weight = NULL;
    w->d_rms_ffn_weight = NULL;
    w->d_wq = NULL; w->d_wk = NULL; w->d_wv = NULL; w->d_wo = NULL;
    w->d_w1 = NULL; w->d_w2 = NULL; w->d_w3 = NULL;
    w->d_rms_final_weight = NULL; w->d_wcls = NULL;

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
}

void cuda_copy_weights_to_device(TransformerWeights* w, Config* p) {
#if USE_CUDA
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
    
    // #ifdef _WIN32
    //     *fd = _open(checkpoint, O_RDONLY);
    //     if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    //     *data = win_mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    //     if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    // #else
        *fd = open(checkpoint, O_RDONLY);
        if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    // #endif

    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
    cuda_copy_weights_to_device(&t->weights, &t->config);
}

void free_transformer(Transformer* t) {
    // #ifdef _WIN32
    //     if (t->data != MAP_FAILED) { win_munmap(t->data, t->file_size); }
    //     if (t->fd != -1) { _close(t->fd); }
    // #else
        if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
        if (t->fd != -1) { close(t->fd); }
    // #endif
    
    free_cuda_weights(&t->weights);
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Fixed hybrid forward pass

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // Copy token embedding
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(s->x, content_row, dim * sizeof(float));

#if USE_CUDA
    CUDA_CHECK(cudaMemcpy(s->d_x, s->x, dim * sizeof(float), cudaMemcpyHostToDevice));
#endif

    const bool profile_this_token = (g_enable_profile && pos == g_profile_pos && !g_profile_triggered);
    float stage1_ms = -1.0f, stage2_ms = -1.0f, stage3_ms = -1.0f, stage4_ms = -1.0f;
    float stage5_ms = -1.0f, stage6_ms = -1.0f, stage7_ms = -1.0f, stage8_ms = -1.0f;
    float total_layer_ms = -1.0f, est_32_layer_ms = -1.0f, est_tok_s = -1.0f;
    cudaEvent_t prof_start = nullptr, prof_stop = nullptr;
    if (profile_this_token) {
        cudaEventCreate(&prof_start);
        cudaEventCreate(&prof_stop);
    }

    // Process all layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        const bool profile_layer0 = profile_this_token && (l == 0);
        
#if USE_CUDA

        if (g_enable_profile && !g_profile_triggered && pos == g_profile_pos) {
            profile_7b_bottlenecks(transformer, token, pos);
            g_profile_triggered = true;
        }
        
        // GPU-accelerated attention
        if (profile_layer0) {
            cudaEventRecord(prof_start);
        }
        cuda_rmsnorm_hybrid(s->d_xb, s->d_x, w->d_rms_att_weight + l*dim, s->xb, dim, l, pos, 1);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage1_ms, prof_start, prof_stop);
            cudaEventRecord(prof_start);
        }

        cuda_matmul(s->d_q, s->d_xb, w->d_wq + l*dim*dim, dim, dim);
        cuda_matmul(s->d_k, s->d_xb, w->d_wk + l*dim*kv_dim, dim, kv_dim);
        cuda_matmul(s->d_v, s->d_xb, w->d_wv + l*dim*kv_dim, dim, kv_dim);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage2_ms, prof_start, prof_stop);
        }

        // Copy back for attention processing (CPU handles attention mechanism)
        auto cpu_stage3_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(s->q, s->d_q, dim * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s->k_original, s->d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(s->v_original, s->d_v, kv_dim * sizeof(float), cudaMemcpyDeviceToHost));
#else
        // CPU fallback
        rmsnorm(s->xb, s->x, w->rms_att_weight + l*dim, dim);
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k_original, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v_original, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
#endif

        // Cache management and attention (CPU)
        int loff = l * p->seq_len * kv_dim;
        float* k_cache_ptr = s->key_cache + loff + pos * kv_dim;
        float* v_cache_ptr = s->value_cache + loff + pos * kv_dim;
        memcpy(k_cache_ptr, s->k_original, kv_dim * sizeof(float));
        memcpy(v_cache_ptr, s->v_original, kv_dim * sizeof(float));
        s->k = k_cache_ptr;
        s->v = v_cache_ptr;

        // RoPE encoding
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        if (profile_layer0) {
            auto cpu_stage3_end = std::chrono::high_resolution_clock::now();
            stage3_ms = std::chrono::duration<float, std::milli>(cpu_stage3_end - cpu_stage3_start).count();
        }

        // Multi-head attention
        auto cpu_stage4_start = std::chrono::high_resolution_clock::now();
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        if (profile_layer0) {
            auto cpu_stage4_end = std::chrono::high_resolution_clock::now();
            stage4_ms = std::chrono::duration<float, std::milli>(cpu_stage4_end - cpu_stage4_start).count();
            cudaEventRecord(prof_start);
        }

#if USE_CUDA
        // Continue with GPU-accelerated FFN

        // 7. CPU to GPU Transfer for FFN
        CUDA_CHECK(cudaMemcpy(s->d_xb, s->xb, dim * sizeof(float), cudaMemcpyHostToDevice));

        // 8. GPU FFN Residual
        cuda_matmul(s->d_xb2, s->d_xb, w->d_wo + l*dim*dim, dim, dim);
        cuda_elementwise_add(s->d_x, s->d_xb2, dim);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage5_ms, prof_start, prof_stop);
            cudaEventRecord(prof_start);
        }
        
        // 9. GPU FFN RMSNorm
        cuda_rmsnorm_hybrid(s->d_xb, s->d_x, w->d_rms_ffn_weight + l*dim, s->xb, dim, l, pos, 0);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage6_ms, prof_start, prof_stop);
            cudaEventRecord(prof_start);
        }

        // 10. GPU FFN W1 and W3 Projections
        cuda_matmul(s->d_hb, s->d_xb, w->d_w1 + l*dim*hidden_dim, dim, hidden_dim);
        cuda_matmul(s->d_hb2, s->d_xb, w->d_w3 + l*dim*hidden_dim, dim, hidden_dim);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage7_ms, prof_start, prof_stop);
            cudaEventRecord(prof_start);
        }

        // 11. GPU SwiGLU Activation
        cuda_swiglu(s->d_hb, s->d_hb2, hidden_dim);

        // 12. GPU FFN W2 Projection + Residual
        cuda_matmul(s->d_xb, s->d_hb, w->d_w2 + l*hidden_dim*dim, hidden_dim, dim);
        cuda_elementwise_add(s->d_x, s->d_xb, dim);

        if (profile_layer0) {
            cudaEventRecord(prof_stop);
            cudaEventSynchronize(prof_stop);
            cudaEventElapsedTime(&stage8_ms, prof_start, prof_stop);
            total_layer_ms = stage1_ms + stage2_ms + stage3_ms + stage4_ms +
                             stage5_ms + stage6_ms + stage7_ms + stage8_ms;
            est_32_layer_ms = total_layer_ms * 32.0f;
            est_tok_s = est_32_layer_ms > 0.0f ? (1000.0f / est_32_layer_ms) : 0.0f;
        }
#else
        // CPU FFN
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2 + l*hidden_dim*dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
#endif
    }

    // Profile metrics collection - triggered at specified position
    if (profile_this_token) {
        g_profile_triggered = true;
        append_profile_metrics_csv(
            token,
            pos,
            p->dim,
            p->hidden_dim,
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
            total_layer_ms,
            est_32_layer_ms,
            est_tok_s);
        printf("Profiling metrics appended to %s\n", g_profile_csv_path);
        if (prof_start) cudaEventDestroy(prof_start);
        if (prof_stop) cudaEventDestroy(prof_stop);
    }

    // Final layer
#if USE_CUDA
    cuda_rmsnorm_hybrid(s->d_x, s->d_x, w->d_rms_final_weight, s->x, dim, 1000, pos, 0);
    cuda_matmul(s->d_logits, s->d_x, w->d_wcls, p->dim, p->vocab_size);
    CUDA_CHECK(cudaMemcpy(s->logits, s->d_logits, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
#else
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);
#endif

    return s->logits;
}

// Include tokenizer, sampler, and main function (same as working debug version)
// [Complete implementation would include all the tokenizer/sampler code here]

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

// void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
//     t->vocab_size = vocab_size;
//     t->vocab = (char**)malloc(vocab_size * sizeof(char*));
//     t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
//     t->sorted_vocab = NULL;
//     for (int i = 0; i < 256; i++) {
//         t->byte_pieces[i * 2] = (unsigned char)i;
//         t->byte_pieces[i * 2 + 1] = '\0';
//     }
//     FILE *file = fopen(tokenizer_path, "rb");
//     if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
//     if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
//     int len;
//     for (int i = 0; i < vocab_size; i++) {
//         if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
//         if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
//         t->vocab[i] = (char *)malloc(len + 1);
//         if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
//         t->vocab[i][len] = '\0';
//     }
//     fclose(file);
// }

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
    // TokenIndex tok = { .str = (char *)str };        // Used a TokenIndex 
    TokenIndex tok;
    tok.str = (char *)str;
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
//     if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

//     if (t->sorted_vocab == NULL) {
//         t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
//         for (int i = 0; i < t->vocab_size; i++) {
//             t->sorted_vocab[i].str = t->vocab[i];
//             t->sorted_vocab[i].id = i;
//         }
//         qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
//     }

//     char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
//     size_t str_len = 0;
//     *n_tokens = 0;
//     if (bos) tokens[(*n_tokens)++] = 1;

//     if (text[0] != '\0') {
//         int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
//         tokens[(*n_tokens)++] = dummy_prefix;
//     }

//     for (char *c = text; *c != '\0'; c++) {
//         if ((*c & 0xC0) != 0x80) {
//             str_len = 0;
//         }
//         str_buffer[str_len++] = *c;
//         str_buffer[str_len] = '\0';
//         if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
//             continue;
//         }
//         int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
//         if (id != -1) {
//             tokens[(*n_tokens)++] = id;
//         } else {
//             for (int i=0; i < str_len; i++) {
//                 tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
//             }
//         }
//         str_len = 0;
//     }

//     while (1) {
//         float best_score = -1e10;
//         int best_id = -1;
//         int best_idx = -1;
//         for (int i=0; i < (*n_tokens-1); i++) {
//             sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
//             int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
//             if (id != -1 && t->vocab_scores[id] > best_score) {
//                 best_score = t->vocab_scores[id];
//                 best_id = id;
//                 best_idx = i;
//             }
//         }
//         if (best_idx == -1) {
//             break;
//         }
//         tokens[best_idx] = best_id;
//         for (int i = best_idx+1; i < (*n_tokens-1); i++) {
//             tokens[i] = tokens[i+1];
//         }
//         (*n_tokens)--;
//     }
//     if (eos) tokens[(*n_tokens)++] = 2;
//     free(str_buffer);
// }

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

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Time utilities

long time_in_ms() {
// #ifdef _WIN32
//     return win_time_in_ms();
// #else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
// #endif
}

// ----------------------------------------------------------------------------
// Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    printf("=== TILED GEMM GPU GENERATION ===\n");
    printf("Using Tiled GEMM with CUDA acceleration\n");
    printf("Steps: %d\n", steps);
    printf("Prompt: \"%s\"\n", prompt ? prompt : "");
    printf("=====================================\n\n");

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
#ifdef DUMP_LOGITS
    int logit_dump_count = 0;
#endif

    printf("Starting generation with token %d (\"%s\")\n", token, tokenizer->vocab[token]);
    printf("Initial prompt tokens: %d\n", num_prompt_tokens);

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
#ifdef DUMP_LOGITS
        if (g_logit_bin_path[0] != '\0' && pos >= num_prompt_tokens - 1) {
            FILE* _lf = fopen(g_logit_bin_path, logit_dump_count == 0 ? "wb" : "ab");
            if (_lf) {
                fwrite(logits, sizeof(float), transformer->config.vocab_size, _lf);
                fclose(_lf);
            }
        }
#endif

        // if (pos < 10) {
        //     printf("\nPrompt: \"%s\"\n", prompt);
        //     printf("Initial token: %d (%s)\n", token, tokenizer->vocab[token]);

            // printf("\n Logits : [ ");
            // for(int i = 0; i < 10; i++) {
            //     printf("%.4f, ", logits[i]);
            // }
            // printf(" ]\n\n");

        // }

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
#ifdef DUMP_LOGITS
        if (g_token_ids_path[0] != '\0' && pos >= num_prompt_tokens - 1) {
            FILE* _tf = fopen(g_token_ids_path, logit_dump_count == 0 ? "w" : "a");
            if (_tf) { fprintf(_tf, logit_dump_count == 0 ? "[%d" : ",%d", next); fclose(_tf); }
            logit_dump_count++;
        }
#endif
        pos++;

        if (next == 1) { printf(" <EOS>\n"); break; }

        char* piece = decode(tokenizer, token, next);
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
#ifdef DUMP_LOGITS
    if (g_token_ids_path[0] != '\0' && logit_dump_count > 0) {
        FILE* _tf = fopen(g_token_ids_path, "a");
        if (_tf) { fprintf(_tf, "]"); fclose(_tf); }
    }
#endif

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
    fprintf(stderr, "  -R <string> profiling CSV output path (default: tiled_profile_metrics.csv)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// Main function

int main(int argc, char *argv[]) {
    // Initialize CUDA
    // printf("=== Assignment 2: Transformer Text Generation (Naive CUDA Approach)===\n");
    printf("Initializing CUDA...\n");
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    printf("CUDA Status: %s\n", cudaGetErrorString(cudaStatus));
    printf("Device Count: %d\n", deviceCount);
    
    // Runtime CUDA availability (this actually works, unlike preprocessor redefinition)
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
            printf("GPU supports large shared memory (>= Ampere architecture)\n");
            if (prop.major == 8 && prop.minor == 0 && prop.multiProcessorCount >= 100) {
                printf("Detected A100 GPU - 163 KB shared memory available\n");
            } else if (prop.major >= 8) {
                printf("Detected RTX 30xx/40xx series - 99 KB shared memory available\n");
            }
        } else if (prop.major == 7) {
            size_t max_shared_mem = (prop.minor == 0) ? 96 : 64;
            printf("Detected RTX 20xx/V100 series - %zu KB shared memory available\n", max_shared_mem);
        }

        printf("Initializing CUDA libraries...\n");
        printf("CUDA initialization complete!\n");
        
        // Validate optimization requirements
        if (prop.sharedMemPerBlock < 48 * 1024) {
            printf("Warning: Limited shared memory may affect performance\n");
        }
        if (prop.major < 7) {
            printf("Warning: GPU compute capability < 7.0 may have reduced performance\n");
        }

    }

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
#ifdef DUMP_LOGITS
        else if (argv[i][1] == 'L') {
            strncpy(g_logit_bin_path, argv[i + 1], sizeof(g_logit_bin_path) - 1);
            g_logit_bin_path[sizeof(g_logit_bin_path) - 1] = '\0';
        }
        else if (argv[i][1] == 'T') {
            strncpy(g_token_ids_path, argv[i + 1], sizeof(g_token_ids_path) - 1);
            g_token_ids_path[sizeof(g_token_ids_path) - 1] = '\0';
        }
#endif
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
    build_transformer(&transformer, checkpoint_path);
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
    if (g_enable_profile)
        printf("  Profiling: enabled (trigger pos=%d, csv=%s)\n", g_profile_pos, g_profile_csv_path);
    printf("\n");

#if USE_CUDA
    if (cuda_available) {
        printf("Tiled GEMM: ENABLED (CUDA mode)\n");
    } else {
        printf("Tiled GEMM: DISABLED (No CUDA device)\n");
    }
#else
    printf("Tiled GEMM: DISABLED (CPU mode)\n");
#endif
    printf("\n");

    // Run generation
    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // Cleanup
    printf("\nCleaning up...\n");
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    
    printf("Done.\n");
    return 0;
}

/*
 * COMPILATION INSTRUCTIONS:
 * 
 * To compile this CUDA-parallelized Llama-2 implementation:
 * 
 * 1. With CUDA support (recommended):
 *    nvcc -o llama2_cuda llama2_cuda.cu -lcuda -lcudart -O3
 * 
 * 2. CPU-only fallback (if CUDA not available):
 *    gcc -o llama2_cpu llama2_cuda.cu -DUSE_CUDA=0 -lm -O3
 * 
 * 3. With additional optimizations:
 *    nvcc -o llama2_cuda llama2_cuda.cu -lcuda -lcudart -O3 -arch=sm_70 --use_fast_math
 * 
 * USAGE:
 *    ./llama2_cuda model.bin -i "Your prompt here" -n 256 -t 1.0 -p 0.9
 * 
 * PERFORMANCE OPTIMIZATIONS IMPLEMENTED:
 * 
 * 1. Matrix Multiplication Parallelization:
 *    - Basic kernel: Each thread computes one output element
 *    - Optimized kernel: Uses shared memory for input vector caching
 *    - Adaptive selection based on matrix dimensions
 * 
 * 2. RMSNorm Parallelization:
 *    - Simple version: Single-thread reduction with broadcast
 *    - Optimized version: Parallel reduction with shared memory
 * 
 * 3. Element-wise Operations:
 *    - Parallelized residual connections
 *    - Parallelized SwiGLU activation function
 * 
 * 4. Memory Management:
 *    - Separate host and device memory pools
 *    - Minimized host-device transfers
 *    - Proper cleanup to prevent memory leaks
 * 
 * 5. Hybrid CPU/GPU Architecture:
 *    - GPU acceleration for compute-intensive operations
 *    - CPU fallback for unsupported operations
 *    - Automatic fallback if CUDA unavailable
 * 
 * PERFORMANCE CHARACTERISTICS:
 * - Significant speedup on matrix operations (typically 5-10x on modern GPUs)
 * - Memory bandwidth optimization through coalesced access patterns
 * - Reduced computation time for large models
 * - Scalable to different GPU architectures
 * 
 * LIMITATIONS AND FUTURE IMPROVEMENTS:
 * 1. Attention mechanism still primarily CPU-based (could be fully GPU-parallelized)
 * 2. RoPE encoding done on CPU (could be parallelized)
 * 3. No batched inference support (could process multiple sequences simultaneously)
 * 4. No mixed-precision support (FP16 could improve performance on modern GPUs)
 * 5. Single-GPU only (could be extended to multi-GPU)
 * 
 * MEMORY REQUIREMENTS:
 * - Host memory: ~2x model size (for weights + activations)
 * - GPU memory: ~1.5x model size (for device weights + activations)
 * - Total: ~3.5x model size in memory
 */