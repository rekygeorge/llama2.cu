#include <stdio.h>
#include <stdint.h>

int main() {
    FILE *file = fopen("llama2_7b.bin", "rb");
    if (!file) {
        printf("Cannot open file\n");
        return 1;
    }
    
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1) {
        printf("Cannot read magic number\n");
        fclose(file);
        return 1;
    }
    
    printf("Magic number in file: 0x%08X\n", magic);
    printf("Expected magic number: 0x616b3432\n");
    printf("Magic as ASCII: ");
    for (int i = 0; i < 4; i++) {
        char c = (magic >> (i * 8)) & 0xFF;
        if (c >= 32 && c <= 126) {
            printf("%c", c);
        } else {
            printf("\\x%02x", (unsigned char)c);
        }
    }
    printf("\n");
    
    fclose(file);
    return 0;
}
