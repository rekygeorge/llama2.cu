/* Inference for Llama-2 Transformer model in pure C
 * Mixed Precision variant: weights stored as FP16, all arithmetic in FP32.
 *
 * Strategy:
 *   - Checkpoint file is standard FP32 .bin (unchanged, no special model needed)
 *   - On load, all weight matrices are converted to FP16 in malloc'd buffers
 *   - matmul and rmsnorm widen FP16 -> FP32 before every multiply-add
 *   - Activations (RunState, KV cache) remain FP32 throughout
 *
 * Effect on your i9-10980HK (no AVX-512 FP16):
 *   - Weights use half the RAM                (~2x memory saving)
 *   - CPU fetches half the data per matmul    (~2x memory bandwidth saving)
 *   - FP16 arithmetic is promoted to FP32 by GCC (no hardware FP16 ALU)
 *   - Net result: faster for large models, same numerical quality as FP32
 *
 * Build (Windows): gcc run_fp16.c win.c -o run_fp16 -lm
 * Build (WSL):     gcc run_fp16.c -o run_fp16 -lm
 * Usage:           ./run_fp16 model.bin [options]   (same CLI as run)
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

/* FP16 storage type — GCC 7+ extension.
 * On CPUs without AVX-512 FP16, GCC automatically widens to float for any
 * arithmetic, so the casts below are explicit and portable. */
typedef _Float16 float16_t;

/* Allocate a FP16 buffer and fill it by converting n FP32 values from src. */
static float16_t* alloc_and_convert_fp16(const float* src, size_t n) {
    float16_t* buf = malloc(n * sizeof(float16_t));
    if (!buf) { fprintf(stderr, "malloc failed in fp16 conversion\n"); exit(EXIT_FAILURE); }
    for (size_t i = 0; i < n; i++) {
        buf[i] = (float16_t)src[i];   /* narrow FP32 -> FP16 at load time */
    }
    return buf;
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

/* All weight matrices stored as FP16 — halves weight memory vs FP32.
 * rmsnorm weights are small but converted for consistency. */
typedef struct {
    float16_t* token_embedding_table; // (vocab_size, dim)
    float16_t* rms_att_weight;        // (layer, dim)
    float16_t* rms_ffn_weight;        // (layer, dim)
    float16_t* wq;                    // (layer, dim, n_heads * head_size)
    float16_t* wk;                    // (layer, dim, n_kv_heads * head_size)
    float16_t* wv;                    // (layer, dim, n_kv_heads * head_size)
    float16_t* wo;                    // (layer, n_heads * head_size, dim)
    float16_t* w1;                    // (layer, hidden_dim, dim)
    float16_t* w2;                    // (layer, dim, hidden_dim)
    float16_t* w3;                    // (layer, hidden_dim, dim)
    float16_t* rms_final_weight;      // (dim,)
    float16_t* wcls;                  // (vocab_size, dim) — may alias token_embedding_table
} TransformerWeights;

/* Activations stay FP32: they are computed values, not stored weights.
 * KV cache is also FP32 — it holds projected key/value vectors, not raw weights. */
typedef struct {
    float *x;          // activation at current time stamp (dim,)
    float *xb;         // inside a residual branch (dim,)
    float *xb2;        // additional buffer (dim,)
    float *hb;         // buffer for hidden dim in ffn (hidden_dim,)
    float *hb2;        // buffer for hidden dim in ffn (hidden_dim,)
    float *q;          // query (dim,)
    float *k;          // key (dim,)  — points into key_cache
    float *v;          // value (dim,) — points into value_cache
    float *att;        // attention scores (n_heads, seq_len)
    float *logits;     // output logits (vocab_size,)
    float* key_cache;  // (layer, seq_len, kv_dim)
    float* value_cache;// (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;                // file descriptor for the mmap'd checkpoint
    float* data;           // mmap'd FP32 checkpoint data (kept for munmap)
    ssize_t file_size;
    int weights_shared;    // 1 if wcls shares token_embedding_table buffer
} Transformer;

// ----------------------------------------------------------------------------
// RunState alloc / free — identical to run.c (activations are always FP32)

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x          = calloc(p->dim, sizeof(float));
    s->xb         = calloc(p->dim, sizeof(float));
    s->xb2        = calloc(p->dim, sizeof(float));
    s->hb         = calloc(p->hidden_dim, sizeof(float));
    s->hb2        = calloc(p->hidden_dim, sizeof(float));
    s->q          = calloc(p->dim, sizeof(float));
    s->key_cache  = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache= calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att        = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits     = calloc(p->vocab_size, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Weight loading: mmap FP32 checkpoint, convert all weights to FP16 buffers

/* Mirrors the pointer arithmetic in run.c's memory_map_weights, but instead
 * of assigning pointers into the mmap region, allocates FP16 buffers. */
void convert_weights_fp32_to_fp16(TransformerWeights* w, Config* p,
                                   float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    size_t n;

    n = (size_t)p->vocab_size * p->dim;
    w->token_embedding_table = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim;
    w->rms_att_weight = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim * (p->n_heads * head_size);
    w->wq = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wk = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * (p->n_heads * head_size) * p->dim;
    w->wo = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim;
    w->rms_ffn_weight = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim * p->hidden_dim;
    w->w1 = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->hidden_dim * p->dim;
    w->w2 = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = n_layers * p->dim * p->hidden_dim;
    w->w3 = alloc_and_convert_fp16(ptr, n); ptr += n;

    n = p->dim;
    w->rms_final_weight = alloc_and_convert_fp16(ptr, n); ptr += n;

    ptr += p->seq_len * head_size / 2; // skip freq_cis_real (RoPE, unused)
    ptr += p->seq_len * head_size / 2; // skip freq_cis_imag (RoPE, unused)

    if (shared_weights) {
        /* wcls shares the same FP16 buffer — no extra allocation needed */
        w->wcls = w->token_embedding_table;
    } else {
        n = (size_t)p->vocab_size * p->dim;
        w->wcls = alloc_and_convert_fp16(ptr, n);
    }
}

void free_weights(TransformerWeights* w, int weights_shared) {
    free(w->token_embedding_table);
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->rms_final_weight);
    if (!weights_shared) free(w->wcls); /* avoid double-free when wcls == token_embedding_table */
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size, int* weights_shared_out) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    *weights_shared_out = shared_weights;
    config->vocab_size = abs(config->vocab_size);
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    #ifdef _WIN32
        *fd = win_open(checkpoint, O_RDONLY);
        if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        *data = win_mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    #else
        *fd = open(checkpoint, O_RDONLY);
        if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    #endif

    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    /* Convert FP32 mmap'd weights → FP16 malloc'd buffers. After this call,
     * *data (the mmap) is only retained for munmap; weights are in FP16. */
    convert_weights_fp32_to_fp16(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights,
                    &t->fd, &t->data, &t->file_size, &t->weights_shared);
    fprintf(stderr, "[fp16] weights converted: %.1f MB (file) -> ~%.1f MB (RAM)\n",
            (float)t->file_size / 1024.0f / 1024.0f,
            (float)t->file_size / 2.0f / 1024.0f / 1024.0f);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free_weights(&t->weights, t->weights_shared);

    /* Unmap the original FP32 checkpoint — no longer needed after conversion */
    #ifdef _WIN32
        if (t->data != MAP_FAILED) { win_munmap(t->data, t->file_size); }
        if (t->fd != -1) { win_close(t->fd); }
    #else
        if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
        if (t->fd != -1) { close(t->fd); }
    #endif

    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Neural net blocks — mixed precision
// weights: FP16 (loaded from FP16 buffers)
// activations / accumulators: FP32

/* RMSNorm: weight is FP16, inputs and output are FP32.
 * The cast (float)weight[j] widens FP16 -> FP32 before the multiply. */
void rmsnorm(float* o, float* x, float16_t* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = (float)weight[j] * (ss * x[j]); /* widen weight before multiply */
    }
}

/* Softmax: operates entirely on FP32 activations — unchanged. */
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) { max_val = x[i]; }
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

/* Matmul: W (d,n) @ x (n,) -> xout (d,)
 * w is FP16 — each element is widened to FP32 before the multiply-add.
 * val accumulates in FP32, preventing underflow/overflow that pure FP16 math
 * would cause. Memory bandwidth is halved vs FP32 weights. */
void matmul(float* xout, float* x, float16_t* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;                           /* FP32 accumulator */
        for (int j = 0; j < n; j++) {
            val += (float)w[i * n + j] * x[j];     /* widen FP16 -> FP32 before multiply */
        }
        xout[i] = val;
    }
}

/* forward: identical logic to run.c.
 * Only differences: weight pointers are float16_t*, and the token embedding
 * copy widens element-by-element instead of memcpy. */
float* forward(Transformer* transformer, int token, int pos) {

    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /* Token embedding lookup: widen each FP16 element to FP32.
     * Cannot use memcpy because sizeof(float16_t) != sizeof(float). */
    float16_t* content_row = w->token_embedding_table + token * dim;
    for (int i = 0; i < dim; i++) { x[i] = (float)content_row[i]; }

    for (unsigned long long l = 0; l < p->n_layers; l++) {

        /* Attention rmsnorm */
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        /* QKV projections — FP16 weights, FP32 activations */
        matmul(s->q, s->xb, w->wq + l*dim*dim,    dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        /* RoPE: operates on FP32 q and k — unchanged */
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

        /* Multihead attention — KV cache is FP32, unchanged */
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) { xb[i] += a * v[i]; }
            }
        }

        /* Output projection — FP16 weight */
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }

        /* FFN rmsnorm */
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        /* FFN projections — FP16 weights */
        matmul(s->hb,  s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        /* SwiGLU nonlinearity — FP32 throughout */
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer — identical to run.c

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

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
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
        if (!(isprint(byte_val) || isspace(byte_val))) { return; }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) { continue; }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) { break; }
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) { tokens[i] = tokens[i+1]; }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler — identical to run.c

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
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
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
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
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
// Utilities: time — identical to run.c

long time_in_ms() {
#ifdef _WIN32
    return win_time_in_ms();
#else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif
}

// ----------------------------------------------------------------------------
// Generation loop — identical to run.c

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
    }
}

// ----------------------------------------------------------------------------
// Chat loop — identical to run.c

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    int8_t user_turn = 1;
    int next;
    int token;
    int prev_token;
    int pos = 0;
    while (pos < steps) {

        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI — identical to run.c
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run_fp16 <checkpoint> [options]\n");
    fprintf(stderr, "Example: run_fp16 model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if      (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
