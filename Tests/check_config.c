#include <stdio.h>
#include <stdint.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

int main() {
    FILE *file = fopen("stories15M.bin", "rb");
    if (!file) {
        printf("Cannot open file\n");
        return 1;
    }
    
    uint32_t header_size;
    if (fread(&header_size, sizeof(uint32_t), 1, file) != 1) {
        printf("Cannot read header size\n");
        fclose(file);
        return 1;
    }
    
    printf("Header size: 0x%08X (%d bytes)\n", header_size, header_size);
    
    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        printf("Cannot read config\n");
        fclose(file);
        return 1;
    }
    
    printf("Model configuration:\n");
    printf("  dim: %d\n", config.dim);
    printf("  hidden_dim: %d\n", config.hidden_dim);
    printf("  n_layers: %d\n", config.n_layers);
    printf("  n_heads: %d\n", config.n_heads);
    printf("  n_kv_heads: %d\n", config.n_kv_heads);
    printf("  vocab_size: %d\n", config.vocab_size);
    printf("  seq_len: %d\n", config.seq_len);
    
    // Calculate memory requirements
    float weight_memory_gb = 0.0f;
    weight_memory_gb += config.vocab_size * config.dim * 4.0f / (1024*1024*1024); // token embeddings
    weight_memory_gb += config.n_layers * config.dim * 4.0f * 3 / (1024*1024*1024); // rmsnorms
    weight_memory_gb += config.n_layers * config.dim * config.dim * 4.0f * 4 / (1024*1024*1024); // attn weights (approx)
    weight_memory_gb += config.n_layers * config.dim * config.hidden_dim * 4.0f * 3 / (1024*1024*1024); // ffn weights
    weight_memory_gb += config.vocab_size * config.dim * 4.0f / (1024*1024*1024); // classifier
    
    printf("Estimated weight memory: %.2f GB\n", weight_memory_gb);
    
    fclose(file);
    return 0;
}
