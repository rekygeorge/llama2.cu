# Universal CUDA Llama-2 Makefile
# Works on Windows Command Prompt, PowerShell, Git Bash, WSL, and Linux

# Compiler settings
NVCC = nvcc
CUDA_FLAGS = -O2 --use_fast_math -arch=sm_75 -std=c++17

# Source files
MAIN_SRC = llama2_flash.cu
DEBUG_SRC = llama2_debug.cu

# Auto-detect environment and shell
ifdef WINDIR
    # We're on Windows, but need to detect PowerShell vs Command Prompt
    POWERSHELL_CHECK := $(shell powershell -Command "Write-Output 'PS'" 2>nul)
    ifeq ($(POWERSHELL_CHECK),PS)
        # PowerShell environment
        EXE_EXT = .exe
        LIBS = -lcublas -lcublasLt -lcurand -lcudart
        EXTRA_SRC = win.c
        PLATFORM = Windows-PowerShell
        RM_CMD = powershell -Command "Remove-Item -Path"
        RM_FLAGS = -ErrorAction SilentlyContinue
        ECHO_CMD = powershell -Command "Write-Host"
        NULL_REDIRECT = 2>$null
        TEST_CMD = powershell -Command "Test-Path"
    else
        # Command Prompt environment
        EXE_EXT = .exe
        LIBS = -lcublas -lcublasLt -lcurand -lcudart
        EXTRA_SRC = win.c
        PLATFORM = Windows-CMD
        RM_CMD = del /Q
        RM_FLAGS = 
        ECHO_CMD = echo
        NULL_REDIRECT = 2>nul
        TEST_CMD = if exist
    endif
else ifdef OS
    ifeq ($(OS),Windows_NT)
        # Windows with Unix-like shell (Git Bash, etc.)
        EXE_EXT = .exe
        LIBS = -lcublas -lcublasLt -lcurand -lcudart
        EXTRA_SRC = win.c
        PLATFORM = Windows-Unix-Shell
        RM_CMD = rm -f
        RM_FLAGS = 
        ECHO_CMD = echo
        NULL_REDIRECT = 2>/dev/null
        TEST_CMD = test -f
    else
        # Unix/Linux
        EXE_EXT = 
        LIBS = -lcublas -lcublasLt -lcurand -lcudart -lm
        EXTRA_SRC = 
        PLATFORM = Unix
        RM_CMD = rm -f
        RM_FLAGS = 
        ECHO_CMD = echo
        NULL_REDIRECT = 2>/dev/null
        TEST_CMD = test -f
    endif
else
    # Default to Unix
    EXE_EXT = 
    LIBS = -lcublas -lcublasLt -lcurand -lcudart -lm
    EXTRA_SRC = 
    PLATFORM = Unix
    RM_CMD = rm -f
    RM_FLAGS = 
    ECHO_CMD = echo
    NULL_REDIRECT = 2>/dev/null
    TEST_CMD = test -f
endif

# Target executables
MAIN_TARGET = llama2_flash$(EXE_EXT)
DEBUG_TARGET = llama2_debug$(EXE_EXT)

# Default target
all: main

# Build main program
main:
	@$(ECHO_CMD) "Building main CUDA program for $(PLATFORM)..."
	$(NVCC) $(CUDA_FLAGS) $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)
	@$(ECHO_CMD) "Build complete: $(MAIN_TARGET)"

# Build debug program  
debug:
	@$(ECHO_CMD) "Building debug CUDA program..."
	$(NVCC) $(CUDA_FLAGS) $(DEBUG_SRC) $(EXTRA_SRC) -o $(DEBUG_TARGET) $(LIBS)
	@$(ECHO_CMD) "Build complete: $(DEBUG_TARGET)"

# Build both programs
both: main debug
	@$(ECHO_CMD) "Both programs built successfully!"

# Optimized build
optimized:
	@$(ECHO_CMD) "Building optimized version..."
	$(NVCC) -O3 --use_fast_math -arch=sm_75 $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)
	@$(ECHO_CMD) "Optimized build complete!"

# Architecture-specific builds
sm_70:
	@$(ECHO_CMD) "Building for SM 7.0 (Volta)..."
	$(NVCC) -O2 --use_fast_math -arch=sm_70 $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)
	@$(ECHO_CMD) "SM 7.0 build complete!"

sm_75:
	@$(ECHO_CMD) "Building for SM 7.5 (Turing RTX 20 series)..."
	$(NVCC) -O2 --use_fast_math -arch=sm_75 $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)
	@$(ECHO_CMD) "SM 7.5 build complete - optimized for RTX 2070 Super!"

sm_80:
	@$(ECHO_CMD) "Building for SM 8.0 (Ampere RTX 30 series)..."
	$(NVCC) -O2 --use_fast_math -arch=sm_80 $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)
	@$(ECHO_CMD) "SM 8.0 build complete!"

# Check CUDA installation with shell-specific commands
cuda-check:
	@$(ECHO_CMD) "Checking CUDA installation..."
	$(NVCC) --version
	@$(ECHO_CMD) "Checking GPU information..."
ifeq ($(PLATFORM),Windows-PowerShell)
	-powershell -Command "try { nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits } catch { Write-Host 'nvidia-smi not available' }"
else ifeq ($(PLATFORM),Windows-CMD)
	-nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>nul || echo nvidia-smi not available
else
	-nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>/dev/null || $(ECHO_CMD) "nvidia-smi not available"
endif

# Test compilation with shell-specific cleanup
test:
	@$(ECHO_CMD) "Testing compilation..."
	$(NVCC) --version
	@$(ECHO_CMD) "Compiling test object..."
	$(NVCC) -c $(MAIN_SRC) -o test.o $(LIBS)
	@$(ECHO_CMD) "Test compilation successful!"
ifeq ($(PLATFORM),Windows-PowerShell)
	-powershell -Command "Remove-Item -Path 'test.o' -ErrorAction SilentlyContinue"
else ifeq ($(PLATFORM),Windows-CMD)
	-del test.o 2>nul
else
	-rm -f test.o
endif

# Run programs
run: $(MAIN_TARGET)
	@$(ECHO_CMD) "Running $(MAIN_TARGET)..."
	./$(MAIN_TARGET) stories15M.bin -i "Once upon a time" -n 50

run-debug: $(DEBUG_TARGET)
	@$(ECHO_CMD) "Running $(DEBUG_TARGET)..."
	./$(DEBUG_TARGET) stories15M.bin -i "Test" -n 10

# Clean build files with shell-specific commands
clean:
	@$(ECHO_CMD) "Cleaning build files..."
ifeq ($(PLATFORM),Windows-PowerShell)
	-powershell -Command "Remove-Item -Path '$(MAIN_TARGET)' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '$(DEBUG_TARGET)' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path 'llama2_flash.exe' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '*.o' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '*.obj' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '*.exp' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '*.lib' -ErrorAction SilentlyContinue"
	-powershell -Command "Remove-Item -Path '*.pdb' -ErrorAction SilentlyContinue"
else ifeq ($(PLATFORM),Windows-CMD)
	-del /Q $(MAIN_TARGET) $(DEBUG_TARGET) *.o *.obj *.exp *.lib *.pdb 2>nul
else
	-$(RM_CMD) $(MAIN_TARGET) $(DEBUG_TARGET) *.o *.obj *.exp *.lib *.pdb 2>/dev/null
endif
	@$(ECHO_CMD) "Clean complete!"

# Advanced clean with verification
clean-verify:
	@$(ECHO_CMD) "Cleaning with verification..."
ifeq ($(PLATFORM),Windows-PowerShell)
	-powershell -Command "if (Test-Path '$(MAIN_TARGET)') { Remove-Item '$(MAIN_TARGET)'; Write-Host 'Removed $(MAIN_TARGET)' }"
	-powershell -Command "if (Test-Path '$(DEBUG_TARGET)') { Remove-Item '$(DEBUG_TARGET)'; Write-Host 'Removed $(DEBUG_TARGET)' }"
	-powershell -Command "Get-ChildItem '*.o', '*.obj', '*.exp', '*.lib', '*.pdb' | Remove-Item; Write-Host 'Removed object files'"
else ifeq ($(PLATFORM),Windows-CMD)
	-if exist $(MAIN_TARGET) del $(MAIN_TARGET) && echo Removed $(MAIN_TARGET)
	-if exist $(DEBUG_TARGET) del $(DEBUG_TARGET) && echo Removed $(DEBUG_TARGET)
	-if exist *.o del *.o && echo Removed .o files
	-if exist *.obj del *.obj && echo Removed .obj files
	-if exist *.exp del *.exp && echo Removed .exp files
	-if exist *.lib del *.lib && echo Removed .lib files
	-if exist *.pdb del *.pdb && echo Removed .pdb files
else
	-$(RM_CMD) $(MAIN_TARGET) $(DEBUG_TARGET) *.o *.obj *.exp *.lib *.pdb
endif
	@$(ECHO_CMD) "Clean with verification complete!"

# Check what files exist before cleaning
list-files:
	@$(ECHO_CMD) "Listing build files..."
ifeq ($(PLATFORM),Windows-PowerShell)
	-powershell -Command "Get-ChildItem -Name '*.exe', '*.o', '*.obj', '*.lib', '*.pdb', '*.exp' | ForEach-Object { Write-Host $_ }"
else ifeq ($(PLATFORM),Windows-CMD)
	-dir /B *.exe *.o *.obj *.lib *.pdb *.exp 2>nul || echo No build files found
else
	-ls -la *.exe *.o *.obj *.lib *.pdb *.exp 2>/dev/null || echo "No build files found"
endif

# Show build information
info:
	@$(ECHO_CMD) "====================================="
	@$(ECHO_CMD) "CUDA Llama-2 Build Information"
	@$(ECHO_CMD) "====================================="
	@$(ECHO_CMD) "Platform: $(PLATFORM)"
	@$(ECHO_CMD) "Compiler: $(NVCC)"
	@$(ECHO_CMD) "Flags: $(CUDA_FLAGS)"
	@$(ECHO_CMD) "Main Source: $(MAIN_SRC)"
	@$(ECHO_CMD) "Debug Source: $(DEBUG_SRC)"
	@$(ECHO_CMD) "Extra Source: $(EXTRA_SRC)"
	@$(ECHO_CMD) "Libraries: $(LIBS)"
	@$(ECHO_CMD) "Main Target: $(MAIN_TARGET)"
	@$(ECHO_CMD) "Debug Target: $(DEBUG_TARGET)"
	@$(ECHO_CMD) "====================================="

# Quick build without verbose output
quick:
	$(NVCC) $(CUDA_FLAGS) $(MAIN_SRC) $(EXTRA_SRC) -o $(MAIN_TARGET) $(LIBS)

# Force rebuild
rebuild: clean main

# Simple help
help:
	@$(ECHO_CMD) "CUDA Llama-2 Makefile Help"
	@$(ECHO_CMD) "=========================="
	@$(ECHO_CMD) "make           - Build main program"
	@$(ECHO_CMD) "make main      - Build main program"
	@$(ECHO_CMD) "make debug     - Build debug program"
	@$(ECHO_CMD) "make both      - Build both programs"
	@$(ECHO_CMD) "make optimized - Build optimized version"
	@$(ECHO_CMD) "make sm_70     - Build for Tesla V100"
	@$(ECHO_CMD) "make sm_75     - Build for RTX 20 series (recommended)"
	@$(ECHO_CMD) "make sm_80     - Build for RTX 30 series"
	@$(ECHO_CMD) "make cuda-check - Check CUDA installation"
	@$(ECHO_CMD) "make test      - Test compilation"
	@$(ECHO_CMD) "make run       - Build and run main program"
	@$(ECHO_CMD) "make run-debug - Build and run debug program"
	@$(ECHO_CMD) "make clean-verify - Clean with verification"
	@$(ECHO_CMD) "make list-files   - List build files"
	@$(ECHO_CMD) "make info      - Show build information"
	@$(ECHO_CMD) "make quick     - Quick build"
	@$(ECHO_CMD) "make help      - Show this help"

# Phony targets
.PHONY: all main debug both optimized sm_70 sm_75 sm_80 cuda-check test run run-debug clean clean-verify list-files info quick rebuild help