#!/usr/bin/env python3
"""
Modal CUDA Llama2 Application with AWS S3 Support
================================================

This application compiles and runs the CUDA implementation of Llama2.c on Modal cloud platform.
It automatically sets up the CUDA environment, compiles the code, downloads required models from S3,
and runs inference with support for multiple model sizes.

Usage:
    modal run modal_app.py
    modal run modal_app.py --model llama2_7b.bin --prompt "Your custom prompt here"
    modal run modal_app.py --model stories    compile_cmd = [
        "nvcc", 
        "-O3", 
        f"-arch={cuda_arch}",       # Dynamic architecture based on GPU
        "-std=c++17",
        "--use_fast_math",          # CRITICAL: Enables fast math functions
        "--ptxas-options=-v",       # Provides compilation feedback
        # "-maxrregcount=64",         # Optimizes register usage for better occupancy
        "-o", "llama2_flash",  # Output executable name
        "llama2_flash.cu",
        "-lcudart"
        # For large models, we might need cuBLAS for better performance
        # Large model issues may be due to our primitive matrix operations being too slow
    ]
    
    # For large models (>1GB), add cuBLAS for better performance
    if model_config['size_gb'] > 1.0:
        compile_cmd.append("-lcublas")
        print(f"⚡ Large model detected: Adding cuBLAS support for better performance")ps 512 --temperature 0.8
    
Environment Variables (for S3 access):
    AWS_ACCESS_KEY_ID: Your AWS access key
    AWS_SECRET_ACCESS_KEY: Your AWS secret key
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
    S3_BUCKET_NAME: Your S3 bucket name
"""

import subprocess
import os
import sys
import time
from pathlib import Path
import modal

# Create Modal app
app = modal.App("llama2-cuda")

# Create persistent volume for model storage
model_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Define CUDA image with all necessary dependencies including AWS CLI
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .env({"DEBIAN_FRONTEND": "noninteractive"})  # Prevent interactive prompts
    .run_commands([
        # Set timezone non-interactively
        "ln -fs /usr/share/zoneinfo/UTC /etc/localtime",
        "echo 'UTC' > /etc/timezone",
    ])
    .apt_install([
        "wget",
        "curl", 
        "build-essential",
        "cmake",
        "git",
        "vim",
        "htop",
        "nvtop",
        "awscli",  # AWS CLI for S3 access
        "unzip",   # For extracting if needed
        "gdb",     # GNU debugger for core dump analysis
        "strace",  # System call tracer
        "valgrind" # Memory debugger (works on CPU code)
    ])
    .pip_install([
        "numpy>=1.24.0",
        "requests>=2.31.0", 
        "tqdm>=4.65.0",
        "boto3>=1.26.0",  # AWS SDK for Python
        "botocore>=1.29.0"
    ])
    .run_commands([
        # Verify CUDA installation
        "nvcc --version",
        # Set up environment
        "echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc",
        "echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc",
        # Install CUDA debugging tools
        "apt-get update && apt-get install -y cuda-tools-12-8 || echo 'CUDA tools installation failed'",
        # Enable core dumps for debugging
        "echo 'ulimit -c unlimited' >> ~/.bashrc",
        "echo 'kernel.core_pattern = core.%p.%t' >> /etc/sysctl.conf || true"
    ])
    # Add local source files to the image
    .add_local_dir(".", remote_path="/app")
)

# Model configurations with S3 paths
MODEL_CONFIGS = {
    "stories15M.bin": {
        "gpu": "A10G",
        "memory": 8192,  # 8GB RAM
        "s3_key": None,  # Will download from HuggingFace
        "download_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "description": "15M parameter model (small, fast)",
        "size_gb": 0.06
    },
    "llama2_7b.bin": {
        "gpu": "A100",  # A100 for large model
        "memory": 32768,  # 32GB RAM for large model
        "s3_key": "llama2_7b.bin",  # S3 path to the model
        "download_url": None,  # S3 only
        "description": "7B parameter model (large, high quality)",
        "size_gb": 26.0
    }
}

# Default S3 configuration (can be overridden by environment variables)
DEFAULT_S3_CONFIG = {
    "bucket_name": "llama2-model",  # Change this to your bucket
    "region": "us-west-1"
}

CUDA_SOURCE_VARIANTS = {
    "cublas": {
        "source_file": "llama2_cublas.cu",
        "binary_name": "llama2_cublas",
    },
    "flash": {
        "source_file": "llama2_flash.cu",
        "binary_name": "llama2_flash",
    },
    "tiled": {
        "source_file": "llama2_tiled.cu",
        "binary_name": "llama2_tiled",
    },
    "cublas_fp16": {
        "source_file": "llama2_cublas_fp16.cu",
        "binary_name": "llama2_cublas_fp16",
    },
}

def get_model_config(model_name):
    """Get configuration for specified model."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    else:
        # Default configuration for unknown models
        return {
            "gpu": "A10G",
            "memory": 8192,
            "s3_key": f"models/{model_name}",  # Assume it's in S3
            "download_url": None,
            "description": "Custom model",
            "size_gb": 1.0
        }

def setup_aws_credentials():
    """Setup AWS credentials from environment variables."""
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_S3_CONFIG["region"])
    
    if not aws_access_key or not aws_secret_key:
        print("⚠️  AWS credentials not found in environment variables")
        print("💡 Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    
    print(f"✅ AWS credentials configured for region: {aws_region}")
    return True

def download_from_s3(s3_key, local_path, bucket_name=None, volume_path="/cache"):
    """Download a file from S3 to Modal volume, with caching support."""
    if not bucket_name:
        bucket_name = os.environ.get("S3_BUCKET_NAME", DEFAULT_S3_CONFIG["bucket_name"])
    
    # Define paths
    volume_file_path = f"{volume_path}/{os.path.basename(local_path)}"
    s3_url = f"s3://{bucket_name}/{s3_key}"
    
    print(f"📦 Checking for cached model: {volume_file_path}")
    
    # Check if file already exists in volume
    if os.path.exists(volume_file_path):
        file_size = os.path.getsize(volume_file_path)
        print(f"✅ Found cached model in volume: {file_size / 1e9:.1f} GB")
        
        # Copy from volume to local working directory
        print(f"📋 Copying from volume to working directory...")
        try:
            import shutil
            shutil.copy2(volume_file_path, local_path)
            print(f"✅ Successfully copied from volume cache")
            return True
        except Exception as e:
            print(f"❌ Failed to copy from volume: {e}")
            # Continue to download if copy fails
    else:
        print(f"⚠️  Model not found in volume cache, will download from S3")
    
    print(f"⬇️  Downloading from S3 to volume: {s3_url}")
    print(f"📁 Volume destination: {volume_file_path}")
    
    try:
        # Ensure volume directory exists
        os.makedirs(os.path.dirname(volume_file_path), exist_ok=True)
        
        # Use AWS CLI to download directly to volume
        cmd = [
            "aws", "s3", "cp",
            s3_url,
            volume_file_path,  # Download to volume instead of local path
            "--no-progress"
        ]
        
        print(f"🔧 S3 download command: {' '.join(cmd)}")
        
        # Start the download process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        # Monitor the download
        for line in process.stdout:
            if line.strip():
                print(f"📥 {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            # Verify file was downloaded to volume
            if os.path.exists(volume_file_path):
                file_size = os.path.getsize(volume_file_path)
                print(f"✅ Successfully downloaded to volume: {file_size / 1e9:.1f} GB")
                
                # Copy from volume to local working directory
                print(f"📋 Copying from volume to working directory...")
                try:
                    import shutil
                    shutil.copy2(volume_file_path, local_path)
                    print(f"✅ Successfully copied to working directory")
                    return True
                except Exception as e:
                    print(f"❌ Failed to copy from volume to working directory: {e}")
                    return False
            else:
                print(f"❌ File not found in volume after download: {volume_file_path}")
                return False
        else:
            print(f"❌ S3 download failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ S3 download error: {e}")
        return False

def download_from_s3_boto3(s3_key, local_path, bucket_name=None, volume_path="/cache"):
    """Alternative S3 download using boto3 with progress callback and volume caching."""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        if not bucket_name:
            bucket_name = os.environ.get("S3_BUCKET_NAME", DEFAULT_S3_CONFIG["bucket_name"])
        
        # Define paths
        volume_file_path = f"{volume_path}/{os.path.basename(local_path)}"
        
        print(f"📦 Checking for cached model: {volume_file_path}")
        
        # Check if file already exists in volume
        if os.path.exists(volume_file_path):
            file_size = os.path.getsize(volume_file_path)
            print(f"✅ Found cached model in volume: {file_size / 1e9:.1f} GB")
            
            # Copy from volume to local working directory
            print(f"📋 Copying from volume to working directory...")
            try:
                import shutil
                shutil.copy2(volume_file_path, local_path)
                print(f"✅ Successfully copied from volume cache")
                return True
            except Exception as e:
                print(f"❌ Failed to copy from volume: {e}")
                # Continue to download if copy fails
        
        print(f"⬇️  Downloading from S3 using boto3: s3://{bucket_name}/{s3_key}")
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # Get object info first
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            file_size = response['ContentLength']
            print(f"📏 File size: {file_size / 1e9:.1f} GB")
        except ClientError as e:
            print(f"❌ Error getting file info: {e}")
            return False
        
        # Ensure volume directory exists
        os.makedirs(os.path.dirname(volume_file_path), exist_ok=True)
        
        # Progress callback
        class ProgressPercentage:
            def __init__(self, size):
                self._size = size
                self._seen_so_far = 0
                
            def __call__(self, bytes_amount):
                self._seen_so_far += bytes_amount
                percentage = (self._seen_so_far / self._size) * 100
                if self._seen_so_far % (100 * 1024 * 1024) == 0:  # Print every 100MB
                    print(f"📥 Progress: {self._seen_so_far / 1e9:.1f} GB ({percentage:.1f}%)")
        
        progress = ProgressPercentage(file_size)
        
        # Download to volume first
        s3_client.download_file(bucket_name, s3_key, volume_file_path, Callback=progress)
        
        print(f"✅ Successfully downloaded to volume")
        
        # Copy from volume to local working directory
        print(f"📋 Copying from volume to working directory...")
        try:
            import shutil
            shutil.copy2(volume_file_path, local_path)
            print(f"✅ Successfully copied to working directory")
            return True
        except Exception as e:
            print(f"❌ Failed to copy from volume to working directory: {e}")
            return False
        
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return False
    except ClientError as e:
        print(f"❌ AWS client error: {e}")
        return False
    except Exception as e:
        print(f"❌ boto3 download error: {e}")
        return False

def _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed):
    """Save output to the cache volume with execution summary"""
    try:
        import time
        from datetime import datetime
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Prepare file path in cache volume
        cache_output_path = f"/cache/{output_file}"
        
        # Create summary information
        summary_info = [
            "=" * 80,
            "CUDA LLAMA2 EXECUTION SUMMARY",
            "=" * 80,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Model: {model}",
            f"Prompt: {prompt}",
            f"Steps: {steps}",
            f"Temperature: {temperature}",
            f"Top-p: {topp}",
            f"Seed: {seed}",
            f"Execution Time: {execution_time:.2f} seconds",
            "=" * 80,
            "",
            "EXECUTION LOG:",
            ""
        ]
        
        # Combine summary and output
        full_output = summary_info + output_lines
        
        # Write to cache volume
        with open(cache_output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_output))
        
        print(f"💾 Output saved to cache volume: {cache_output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to save output to cache: {e}")
        return False

# Create separate optimized functions for different model types
@app.function(
    image=cuda_image,
    gpu="A10G",      # Smaller GPU for small models
    memory=8192,     # 8GB RAM
    timeout=3600,    # 1 hour
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def compile_and_run_cuda_small(
    model: str = "stories15M.bin",
    prompt: str = "Once upon a time",
    steps: int = 256,
    temperature: float = 1.0,
    topp: float = 0.9,
    seed: int = 42,
    use_boto3: bool = False,
    save_output: bool = True,
    output_file: str = "profiling.txt",
    cuda_impl: str = "cublas",
    profile_enable: bool = False,
    profile_pos: int = 10,
    profile_csv: str = "/cache/flash_profile_metrics.csv"
):
    """Optimized function for small models (< 1GB)."""
    return _compile_and_run_cuda_impl(
        model,
        prompt,
        steps,
        temperature,
        topp,
        seed,
        use_boto3,
        save_output,
        output_file,
        cuda_impl,
        profile_enable,
        profile_pos,
        profile_csv,
    )

@app.function(
    image=cuda_image,
    gpu="A100",      # Powerful GPU for large models
    memory=32768,    # 32GB RAM  
    timeout=7200,    # 2 hours
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def compile_and_run_cuda_large(
    model: str = "llama2_7b.bin",
    prompt: str = "Once upon a time",
    steps: int = 256,
    temperature: float = 1.0,
    topp: float = 0.9,
    seed: int = 42,
    use_boto3: bool = False,
    save_output: bool = True,
    output_file: str = "profiling.txt",
    cuda_impl: str = "cublas",
    profile_enable: bool = False,
    profile_pos: int = 10,
    profile_csv: str = "/cache/flash_profile_metrics.csv"
):
    """Optimized function for large models (> 1GB)."""
    return _compile_and_run_cuda_impl(
        model,
        prompt,
        steps,
        temperature,
        topp,
        seed,
        use_boto3,
        save_output,
        output_file,
        cuda_impl,
        profile_enable,
        profile_pos,
        profile_csv,
    )

def _compile_and_run_cuda_impl(
    model: str = "stories15M.bin",
    prompt: str = "Once upon a time",
    steps: int = 256,
    temperature: float = 1.0,
    topp: float = 0.9,
    seed: int = 42,
    use_boto3: bool = False,
    save_output: bool = True,
    output_file: str = "profiling.txt",
    cuda_impl: str = "cublas",
    profile_enable: bool = False,
    profile_pos: int = 10,
    profile_csv: str = "/cache/flash_profile_metrics.csv"
):
    """
    Compile and run the CUDA Llama2 implementation with S3 model download support.
    
    Args:
        model: Model file to use (stories15M.bin or llama2_7b.bin)
        prompt: Input text prompt for generation
        steps: Number of generation steps
        temperature: Sampling temperature (0.0 = greedy, 1.0 = original)
        topp: Top-p nucleus sampling parameter
        seed: Random seed for reproducibility
        use_boto3: Use boto3 instead of AWS CLI for S3 downloads
        save_output: Whether to save output to profiling file
        output_file: Name of the output file to save in the cache volume
        cuda_impl: CUDA implementation to compile and run ("cublas", "flash", "tiled", or "cublas_fp16")
        profile_enable: Enable per-layer profiling inside CUDA binary (not supported by cublas_fp16)
        profile_pos: Token position at which profiling is triggered
        profile_csv: CSV output path inside container (use /cache for persistence)
    """
    
    # Initialize output capture
    output_lines = []
    import time
    from datetime import datetime
    
    def log_and_capture(message):
        """Print message and capture it for file output"""
        print(message)
        output_lines.append(message)
    
    # Record start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Change to the app directory where files are mounted
    os.chdir("/app")
    
    log_and_capture(f"🕐 Execution started: {timestamp}")
    log_and_capture("🚀 Starting CUDA Llama2 compilation and execution with S3 support...")
    log_and_capture(f"📂 Working directory: {os.getcwd()}")
    log_and_capture(f"🤖 Selected model: {model}")

    cuda_impl_key = cuda_impl.strip().lower()
    impl_config = CUDA_SOURCE_VARIANTS.get(cuda_impl_key)
    if impl_config is None:
        valid_impls = ", ".join(sorted(CUDA_SOURCE_VARIANTS.keys()))
        log_and_capture(f"❌ Invalid CUDA implementation '{cuda_impl}'. Valid options: {valid_impls}")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False

    source_file = impl_config["source_file"]
    binary_name = impl_config["binary_name"]
    executable_path = f"/app/{binary_name}"
    log_and_capture(f"🧩 CUDA implementation: {cuda_impl_key} ({source_file})")
    
    # Get model configuration
    model_config = get_model_config(model)
    log_and_capture(f"📊 Model info: {model_config['description']}")
    log_and_capture(f"📏 Expected size: {model_config['size_gb']:.1f} GB")
    
    # List directory contents (helpful for debugging)
    log_and_capture(f"📁 Directory contents: {os.listdir('.')}")
    
    # Setup AWS credentials
    if not setup_aws_credentials():
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    # Verify CUDA installation
    log_and_capture("\n🔍 Verifying CUDA installation...")
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        log_and_capture(f"✅ CUDA Compiler: {result.stdout.strip()}")
        
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        log_and_capture(f"✅ GPU Status:\n{result.stdout}")
        
        # Get detailed GPU information
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        gpu_details = result.stdout.strip().split(', ')
        if len(gpu_details) >= 4:
            log_and_capture(f"🎯 GPU Details:")
            log_and_capture(f"   Name: {gpu_details[0]}")
            log_and_capture(f"   Compute Capability: {gpu_details[1]}")
            log_and_capture(f"   Driver Version: {gpu_details[2]}")
            log_and_capture(f"   Total Memory: {gpu_details[3]} MB")
        
        # Check CUDA runtime version
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        cuda_version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
        if cuda_version_line:
            log_and_capture(f"🔧 CUDA Toolkit: {cuda_version_line[0].strip()}")
        
        # Extract GPU memory info for large model validation
        gpu_memory_lines = [line for line in result.stdout.split('\n') if 'MiB' in line and '/' in line]
        if gpu_memory_lines:
            # Parse memory info from nvidia-smi output
            memory_line = gpu_memory_lines[0]  # First GPU line
            log_and_capture(f"🔍 GPU Memory: {memory_line}")
            
            # Check if we have enough memory for large models
            if model_config['size_gb'] > 20:  # Large model threshold
                log_and_capture(f"⚠️  Large model detected ({model_config['size_gb']:.1f} GB)")
                log_and_capture(f"💾 Ensure adequate GPU memory is available")
                
    except subprocess.CalledProcessError as e:
        log_and_capture(f"❌ CUDA verification failed: {e}")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    # Check for required source files
    required_files = [source_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        log_and_capture(f"❌ Missing required files: {missing_files}")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    log_and_capture("✅ All required source files found")
    
    # Handle model and tokenizer files
    log_and_capture(f"\n📥 Checking model and tokenizer files...")
    
    tokenizer_file = "tokenizer.bin"
    
    # Check if model file exists locally
    if os.path.exists(model):
        log_and_capture(f"✅ Model file {model} found locally")
        file_size = os.path.getsize(model)
        log_and_capture(f"📏 Model size: {file_size / 1e9:.1f} GB")
        
        # Validate model file size against expected size
        expected_size_gb = model_config['size_gb']
        actual_size_gb = file_size / 1e9
        size_diff_percent = abs(actual_size_gb - expected_size_gb) / expected_size_gb * 100
        
        if size_diff_percent > 10:  # Allow 10% variance
            log_and_capture(f"⚠️  WARNING: Model size mismatch!")
            log_and_capture(f"   Expected: {expected_size_gb:.1f} GB")
            log_and_capture(f"   Actual: {actual_size_gb:.1f} GB")
            log_and_capture(f"   Difference: {size_diff_percent:.1f}%")
            log_and_capture(f"   This may indicate a corrupted or incomplete download")
        else:
            log_and_capture(f"✅ Model size validation passed ({size_diff_percent:.1f}% difference)")
    else:
        # Download model
        s3_key = model_config.get("s3_key")
        download_url = model_config.get("download_url")
        
        if s3_key:
            log_and_capture(f"📦 Model will be downloaded from S3: {s3_key}")
            # Download from S3 to volume with caching
            if use_boto3:
                success = download_from_s3_boto3(s3_key, model, volume_path="/cache")
            else:
                success = download_from_s3(s3_key, model, volume_path="/cache")
            
            if not success:
                if save_output:
                    _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
                return False
                
        elif download_url:
            log_and_capture(f"⬇️  Downloading {model} from HuggingFace...")
            try:
                subprocess.run(["wget", download_url, "-O", model], check=True)
                log_and_capture(f"✅ Downloaded {model}")
            except subprocess.CalledProcessError as e:
                log_and_capture(f"❌ Failed to download {model}: {e}")
                if save_output:
                    _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
                return False
        else:
            log_and_capture(f"❌ No download source available for {model}")
            if save_output:
                _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
            return False
    
    # Handle tokenizer file
    if not os.path.exists(tokenizer_file):
        log_and_capture(f"⬇️  Downloading {tokenizer_file}...")
        try:
            subprocess.run([
                "wget",
                "https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.bin"
            ], check=True)
            log_and_capture(f"✅ Downloaded {tokenizer_file}")
        except subprocess.CalledProcessError as e:
            log_and_capture(f"❌ Failed to download {tokenizer_file}: {e}")
            if save_output:
                _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
            return False
    else:
        log_and_capture(f"✅ {tokenizer_file} already exists")
    
    # Final validation of both model and tokenizer files
    log_and_capture(f"\n🔍 Final file validation...")
    if not os.path.exists(model) or not os.path.exists(tokenizer_file):
        log_and_capture(f"❌ Required files missing after download attempts:")
        log_and_capture(f"   Model ({model}): {'✅' if os.path.exists(model) else '❌'}")
        log_and_capture(f"   Tokenizer ({tokenizer_file}): {'✅' if os.path.exists(tokenizer_file) else '❌'}")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    # Basic header validation for model files (optional but helpful)
    try:
        log_and_capture(f"🔍 Validating model file format...")
        with open(model, 'rb') as f:
            header = f.read(32)  # Read first 32 bytes
            log_and_capture(f"📄 Model file header (first 32 bytes): {header[:16].hex()}")
            
            # Validate magic number for Llama models
            # Many Llama models start with specific magic numbers
            magic = int.from_bytes(header[:4], byteorder='little')
            log_and_capture(f"🔮 Magic number: {magic} (0x{magic:08x})")
            
            # Check if this looks like a valid model file
            if magic == 0x67676d6c:  # "lmgg" in little endian
                log_and_capture(f"✅ Detected GGML format model")
            elif header[:4] == b'\x20\x01\x00\x00':  # Common header pattern
                log_and_capture(f"✅ Detected Llama-style model header")
            else:
                log_and_capture(f"⚠️  Unknown model format, magic: {header[:8].hex()}")
                
    except Exception as e:
        log_and_capture(f"⚠️  Could not read model header: {e}")
    
    log_and_capture(f"✅ All files validated successfully")
    
    # Compile CUDA code
    log_and_capture("\n🔨 Compiling CUDA code...")
    
    # Set CUDA environment variables
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"  # Increase concurrent kernel connections
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"          # Enable async kernel launches
    
    # Configure shared memory to 96KB per block
    log_and_capture(f"🔧 Configuring CUDA shared memory to 96KB per block...")
    
    # Compilation command - use sm_80 for A100
    # compile_cmd = [
    #     "nvcc", 
    #     "-O3", 
    #     "-arch=sm_80",  # sm_80 for A100 GPU
    #     "-std=c++14",
    #     "-o", "llama2_flash_optimized",  # Output executable name
    #     "llama2_flash_optimized.cu",
    #     "-lcudart"
    # ]

    # compile_cmd = [
    #     "nvcc", 
    #     "-O3", 
    #     "-arch=sm_80",              # sm_80 for A100 GPU
    #     "-std=c++14",
    #     "--use_fast_math",          # CRITICAL: Enables fast math functions
    #     "--ptxas-options=-v",       # Provides compilation feedback
    #     "-maxrregcount=64",         # Optimizes register usage for better occupancy
    #     "-o", "llama2_flash",  # Output executable name
    #     "llama2_flash.cu",
    #     "-lcudart"
    # ]

    # Get model configuration for dynamic compilation
    model_config = get_model_config(model)
    
    # Determine CUDA architecture based on GPU type
    gpu_arch_map = {
        "A100": "sm_80",    # A100
        "A10G": "sm_86",    # A10G 
        "T4": "sm_75",      # T4
        "V100": "sm_70"     # V100
    }
    
    gpu_type = model_config.get("gpu", "A10G")
    cuda_arch = gpu_arch_map.get(gpu_type, "sm_86")  # Default to sm_86 for A10G
    
    # Try to auto-detect actual GPU compute capability
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        actual_compute_cap = result.stdout.strip()
        if actual_compute_cap:
            # Convert compute capability to SM architecture
            cap_parts = actual_compute_cap.split('.')
            if len(cap_parts) == 2:
                major, minor = cap_parts
                detected_arch = f"sm_{major}{minor}"
                log_and_capture(f"🔍 Auto-detected GPU compute capability: {actual_compute_cap} -> {detected_arch}")
                
                # Use detected architecture if different from configured
                if detected_arch != cuda_arch:
                    log_and_capture(f"⚠️  GPU architecture mismatch! Configured: {cuda_arch}, Detected: {detected_arch}")
                    log_and_capture(f"🔧 Using detected architecture: {detected_arch}")
                    cuda_arch = detected_arch
    except Exception as e:
        log_and_capture(f"⚠️  Could not auto-detect GPU compute capability: {e}")
    
    log_and_capture(f"🎯 Detected GPU type: {gpu_type}, using CUDA architecture: {cuda_arch}")

    # Base compilation command
    compile_cmd = [
        "nvcc", 
        "-O3", 
        f"-arch={cuda_arch}",       # Dynamic architecture based on GPU
        "-std=c++17",
        "--use_fast_math",          # CRITICAL: Enables fast math functions
        "--ptxas-options=-v",       # Provides compilation feedback
        "-maxrregcount=64",         # Optimizes register usage for better occupancy
        "-Xptxas", "-maxrregcount=64", # Set max registers per thread
        "-Xptxas", "--opt-level=3", # Additional PTX optimization
        "--default-stream", "per-thread", # Enable per-thread default stream
        "-DMAX_SHARED_MEMORY_PER_BLOCK=98304"  # Set 96KB (98304 bytes) shared memory
    ]
    
    # Add A100-specific optimization flags
    if gpu_type == "A100":
        log_and_capture(f"⚡ A100 detected: Adding tensor core and advanced optimization flags")
        compile_cmd.extend([
            "-DUSE_TENSOR_CORES",   # Enable tensor core usage
            "-Xptxas", "-O3",       # Additional PTX assembler optimization
        ])
    
    # Add output file and source
    compile_cmd.extend([
        "-o", binary_name,           # Output executable name
        source_file,                  # Source file
        "-lcudart",                 # CUDA runtime library
        "-lcublas"                  # cuBLAS library for optimized linear algebra
    ])
    
    log_and_capture(f"🔧 Compilation command: {' '.join(compile_cmd)}")
    
    # Try compilation with fallback architectures if needed
    compilation_successful = False
    fallback_architectures = [cuda_arch, "sm_80", "sm_86", "sm_75", "sm_70"]  # Reorder to try A100 first for large models
    
    # For large models, prioritize A100 architecture
    if model_config['size_gb'] > 1.0:
        fallback_architectures = ["sm_80", "sm_86", "sm_75", "sm_70"]  # A100 first for large models
    
    for attempt, arch in enumerate(fallback_architectures):
        if compilation_successful:
            break
            
        # Update architecture in command
        current_cmd = compile_cmd.copy()
        for i, arg in enumerate(current_cmd):
            if arg.startswith("-arch="):
                current_cmd[i] = f"-arch={arch}"
                break
        
        if attempt > 0:
            log_and_capture(f"🔄 Retry #{attempt} with architecture: {arch}")
            log_and_capture(f"🔧 Fallback command: {' '.join(current_cmd)}")
        
        try:
            result = subprocess.run(
                current_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd="/app"
            )
            log_and_capture("✅ CUDA compilation successful!")
            final_cuda_arch = arch  # Store the successful architecture
            if result.stdout:
                log_and_capture(f"📝 Compiler output: {result.stdout}")
            if result.stderr:
                log_and_capture(f"⚠️  Compiler warnings: {result.stderr}")
            compilation_successful = True
            if attempt > 0:
                log_and_capture(f"🎯 Successfully compiled with fallback architecture: {arch}")
                
        except subprocess.CalledProcessError as e:
            log_and_capture(f"❌ Compilation attempt {attempt + 1} failed with {arch}!")
            log_and_capture(f"🔥 Error code: {e.returncode}")
            log_and_capture(f"📝 stdout: {e.stdout}")
            log_and_capture(f"📝 stderr: {e.stderr}")
            
            if attempt == len(fallback_architectures) - 1:  # Last attempt
                log_and_capture(f"💥 All compilation attempts failed!")
                if save_output:
                    _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
                return False
    
    if not compilation_successful:
        log_and_capture(f"💥 Compilation failed with all attempted architectures!")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    # Verify executable was created
    if not os.path.exists(executable_path):
        log_and_capture(f"❌ Executable not found at {executable_path}")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        return False
    
    log_and_capture(f"✅ Executable created: {executable_path}")
    
    # Get executable info
    try:
        result = subprocess.run(["file", executable_path], capture_output=True, text=True, check=True)
        log_and_capture(f"📋 Executable info: {result.stdout.strip()}")
        
        result = subprocess.run(["ldd", executable_path], capture_output=True, text=True, check=True)
        log_and_capture(f"📚 Library dependencies:")
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                log_and_capture(f"   {line.strip()}")
                
    except subprocess.CalledProcessError as e:
        log_and_capture(f"⚠️  Could not get executable info: {e}")
    
    # Make executable if needed
    os.chmod(executable_path, 0o755)
    
    # Run the CUDA Llama2 inference
    log_and_capture(f"\n🚀 Running CUDA Llama2 inference...")
    log_and_capture(f"🤖 Model: {model}")
    log_and_capture(f"📝 Prompt: '{prompt}'")
    log_and_capture(f"⚙️  Steps: {steps}, Temperature: {temperature}, Top-p: {topp}, Seed: {seed}")
    if profile_enable:
        log_and_capture(f"📈 Profiling enabled: trigger_pos={profile_pos}, csv={profile_csv}")
    
    # Add model size information for debugging large model issues
    model_size_gb = os.path.getsize(model) / 1e9
    log_and_capture(f"📏 Actual model size being executed: {model_size_gb:.1f} GB")
    
    if model_size_gb > 20:  # Large model warning
        log_and_capture(f"⚠️  Large model detected - monitoring for potential issues...")
        log_and_capture(f"💾 GPU memory should be sufficient for A100 (40GB+ available)")
    
    run_cmd = [
        f"./{binary_name}",
        model,  # Use selected model
        "-z", tokenizer_file,
        "-i", prompt,
        "-n", str(steps),
        "-t", str(temperature),
        "-p", str(topp),
        "-s", str(seed)
    ]
    if profile_enable:
        # run_fp16.cu (cublas_fp16) does not implement -P/-R profiling flags.
        # The other implementations (cublas, flash, tiled) do support them.
        if cuda_impl_key == "cublas_fp16":
            log_and_capture("⚠️  Per-layer profiling (-P/-R flags) is not supported by run_fp16.cu. Ignoring --profile-enable for cublas_fp16.")
        else:
            run_cmd.extend(["-P", str(profile_pos), "-R", profile_csv])
    
    log_and_capture(f"🔧 Run command: {' '.join(run_cmd)}")
    
    # Add environment checks before execution
    log_and_capture(f"\n🔬 Pre-execution environment check:")
    log_and_capture(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    log_and_capture(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    log_and_capture(f"   PATH: {os.environ.get('PATH', 'Not set')[:200]}...")
    
    # Check CUDA runtime availability
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        log_and_capture(f"   Available GPUs: {result.stdout.strip()}")
    except:
        log_and_capture(f"   ⚠️  Could not list GPUs")
    
    # Try execution with multiple strategies to handle segmentation faults
    execution_successful = False
    execution_strategies = [
        {
            "name": "Standard execution with 96KB shared memory",
            "cmd": run_cmd,
            "env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
            "timeout": 300
        },
        {
            "name": "Reduced batch size execution",
            "cmd": run_cmd[:-2] + ["-n", str(min(steps, 50))],  # Reduce steps
            "env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
            "timeout": 120
        },
        {
            "name": "Conservative CUDA execution",
            "cmd": run_cmd,
            "env_vars": {"CUDA_LAUNCH_BLOCKING": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_DEVICE_MAX_CONNECTIONS": "32"},
            "timeout": 180
        },
        {
            "name": "Fallback GPU execution",
            "cmd": run_cmd,
            "env_vars": {"CUDA_VISIBLE_DEVICES": "0", "CUDA_CACHE_DISABLE": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "32"},
            "timeout": 180
        }
    ]
    
    for strategy_num, strategy in enumerate(execution_strategies):
        if execution_successful:
            break
            
        current_cmd = strategy["cmd"]
        strategy_env = os.environ.copy()
        strategy_env.update(strategy["env_vars"])
        
        log_and_capture(f"\n🔄 Execution Strategy #{strategy_num + 1}: {strategy['name']}")
        if strategy["env_vars"]:
            log_and_capture(f"   Environment: {strategy['env_vars']}")
        log_and_capture(f"   Command: {' '.join(current_cmd)}")
        log_and_capture(f"   Timeout: {strategy['timeout']}s")

        try:
            # Run with real-time output and enhanced error handling
            log_and_capture("\n" + "="*60)
            log_and_capture(f"🤖 LLAMA2 CUDA OUTPUT ({model}) - Strategy {strategy_num + 1}:")
            log_and_capture("="*60)
            
            # Add timeout and better process handling
            process = subprocess.Popen(
                current_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Separate stderr for better error handling
                text=True,
                cwd="/app",
                bufsize=1,
                universal_newlines=True,
                env=strategy_env  # Use strategy-specific environment
            )
            
            # Stream output in real-time and capture it
            stdout_lines = []
            stderr_lines = []
            
            # Read both stdout and stderr
            import select
            import sys
            
            while process.poll() is None:
                # Check if there's data to read
                try:
                    # Read stdout
                    if process.stdout:
                        line = process.stdout.readline()
                        if line:
                            line = line.rstrip()
                            print(line)  # Still print to console
                            output_lines.append(line)  # Capture for file
                            stdout_lines.append(line)
                    
                    # Read stderr
                    if process.stderr:
                        err_line = process.stderr.readline()
                        if err_line:
                            err_line = err_line.rstrip()
                            print(f"STDERR: {err_line}")  # Print stderr with prefix
                            output_lines.append(f"STDERR: {err_line}")
                            stderr_lines.append(err_line)
                            
                except Exception as e:
                    log_and_capture(f"⚠️  Error reading process output: {e}")
                    break
            
            # Read any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                for line in remaining_stdout.strip().split('\n'):
                    if line:
                        print(line)
                        output_lines.append(line)
                        stdout_lines.append(line)
            
            if remaining_stderr:
                for line in remaining_stderr.strip().split('\n'):
                    if line:
                        print(f"STDERR: {line}")
                        output_lines.append(f"STDERR: {line}")
                        stderr_lines.append(line)
            
            return_code = process.returncode
            log_and_capture("="*60)
            log_and_capture(f"🔍 Process completed with return code: {return_code}")
            
            # Check if execution was successful
            if return_code == 0:
                log_and_capture(f"✅ Strategy {strategy_num + 1} succeeded!")
                execution_successful = True
                break  # Exit the strategy loop on success
            else:
                log_and_capture(f"❌ Strategy {strategy_num + 1} failed with return code: {return_code}")
                
                # For segmentation faults, log the specific strategy failure
                if return_code == -11:
                    log_and_capture(f"💥 SEGMENTATION FAULT with {strategy['name']}")
                    
                    # Only do detailed debugging on the last strategy
                    if strategy_num == len(execution_strategies) - 1:
                        log_and_capture(f"🔍 Running detailed analysis (final strategy)...")
                        _run_detailed_segfault_analysis(
                            log_and_capture,
                            current_cmd,
                            final_cuda_arch if 'final_cuda_arch' in locals() else cuda_arch,
                            executable_path,
                        )
                    else:
                        log_and_capture(f"🔄 Will try next execution strategy...")
                
                # Log stderr content if present
                if stderr_lines:
                    log_and_capture(f"\n❌ STDERR OUTPUT for {strategy['name']}:")
                    for line in stderr_lines[:10]:  # Limit to first 10 lines
                        log_and_capture(f"   {line}")
                    if len(stderr_lines) > 10:
                        log_and_capture(f"   ... ({len(stderr_lines) - 10} more lines)")
        
        except Exception as e:
            log_and_capture(f"❌ Runtime error in {strategy['name']}: {e}")
            
        # If this was the last strategy or we had success, break
        if execution_successful or strategy_num == len(execution_strategies) - 1:
            break
    
    # Final result processing
    if execution_successful:
        log_and_capture(f"🎉 CUDA Llama2 execution completed successfully with {model}!")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        try:
            model_volume.commit()
            log_and_capture("💾 Volume commit completed")
        except Exception as e:
            log_and_capture(f"⚠️  Volume commit failed: {e}")
        return True
    else:
        log_and_capture(f"💥 All execution strategies failed with {model}!")
        if save_output:
            _save_output_to_cache(output_lines, output_file, start_time, model, prompt, steps, temperature, topp, seed)
        try:
            model_volume.commit()
            log_and_capture("💾 Volume commit completed")
        except Exception as e:
            log_and_capture(f"⚠️  Volume commit failed: {e}")
        return False

def _run_detailed_segfault_analysis(log_and_capture, run_cmd, cuda_arch, executable_path):
    """Run detailed segmentation fault analysis - separated for clarity"""
    log_and_capture(f"💥 SEGMENTATION FAULT DETECTED (SIGSEGV)")
    log_and_capture(f"🔍 Common causes on Modal:")
    log_and_capture(f"   1. GPU architecture mismatch (compiled for {cuda_arch}, running on different GPU)")
    log_and_capture(f"   2. CUDA driver/runtime version incompatibility")
    log_and_capture(f"   3. Memory access violation in CUDA kernels")
    log_and_capture(f"   4. Insufficient GPU memory or memory fragmentation")
    log_and_capture(f"   5. Missing or incompatible CUDA libraries")
    
    # First, enable core dumps
    try:
        subprocess.run(["ulimit", "-c", "unlimited"], shell=True, check=False)
        log_and_capture(f"✅ Core dumps enabled")
    except Exception as e:
        log_and_capture(f"⚠️  Could not enable core dumps: {e}")
    
    # Try multiple debugging approaches
    debug_attempts = [
        {
            "name": "cuda-memcheck",
            "cmd": ["cuda-memcheck", "--tool", "memcheck", "--leak-check", "full"] + run_cmd,
            "timeout": 120
        },
        {
            "name": "cuda-gdb (backtrace)",
            "cmd": ["cuda-gdb", "--batch", "--ex", "run", "--ex", "bt", "--ex", "quit", "--args"] + run_cmd,
            "timeout": 60
        },
        {
            "name": "strace (system calls)",
            "cmd": ["strace", "-e", "trace=memory", "-o", "/tmp/strace.out"] + run_cmd,
            "timeout": 30
        }
    ]
    
    for debug_attempt in debug_attempts:
        try:
            log_and_capture(f"🔬 Trying {debug_attempt['name']}: {' '.join(debug_attempt['cmd'][:5])}...")
            debug_result = subprocess.run(
                debug_attempt['cmd'], 
                capture_output=True, 
                text=True, 
                timeout=debug_attempt['timeout'], 
                cwd="/app"
            )
            
            if debug_result.stdout or debug_result.stderr:
                log_and_capture(f"🔍 {debug_attempt['name']} output:")
                if debug_result.stdout:
                    log_and_capture(f"   STDOUT: {debug_result.stdout[:500]}...")
                if debug_result.stderr:
                    log_and_capture(f"   STDERR: {debug_result.stderr[:500]}...")
                break  # Stop after first successful debug attempt
            else:
                log_and_capture(f"⚠️  {debug_attempt['name']} produced no output")
                
        except subprocess.TimeoutExpired:
            log_and_capture(f"⏰ {debug_attempt['name']} timed out after {debug_attempt['timeout']}s")
        except FileNotFoundError:
            log_and_capture(f"⚠️  {debug_attempt['name']} tool not available")
        except Exception as debug_e:
            log_and_capture(f"⚠️  {debug_attempt['name']} failed: {debug_e}")
    
    # Check for strace output
    if os.path.exists("/tmp/strace.out"):
        try:
            with open("/tmp/strace.out", "r") as f:
                strace_content = f.read()
                if strace_content:
                    log_and_capture(f"🔍 System call trace (last 500 chars):")
                    log_and_capture(f"   {strace_content[-500:]}")
        except Exception as e:
            log_and_capture(f"⚠️  Could not read strace output: {e}")
            
    # Check for core dump
    core_files = [f for f in os.listdir("/app") if f.startswith("core")]
    if core_files:
        log_and_capture(f"💀 Core dump files found: {core_files}")
        # Try to get backtrace from core dump
        try:
            core_file = core_files[0]
            gdb_cmd = ["gdb", "--batch", "--ex", "bt", executable_path, f"/app/{core_file}"]
            log_and_capture(f"🔍 Analyzing core dump: {' '.join(gdb_cmd)}")
            gdb_result = subprocess.run(gdb_cmd, capture_output=True, text=True, timeout=30)
            if gdb_result.stdout:
                log_and_capture(f"💀 Core dump backtrace:")
                log_and_capture(f"   {gdb_result.stdout}")
        except Exception as gdb_e:
            log_and_capture(f"⚠️  Could not analyze core dump: {gdb_e}")
    else:
        log_and_capture(f"📝 No core dump files found")
        
    # Additional architecture and library compatibility checks
    log_and_capture(f"\n🔍 Advanced compatibility analysis:")
    
    try:
        # Check actual GPU compute capability vs compiled architecture
        result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        gpu_info = result.stdout.strip().split(', ')
        if len(gpu_info) >= 2:
            actual_cap = gpu_info[0]
            gpu_name = gpu_info[1]
            log_and_capture(f"   🎯 Actual GPU: {gpu_name} (compute {actual_cap})")
            
            # Check if compiled architecture is compatible
            compiled_arch = cuda_arch  # From compilation
            log_and_capture(f"   🔨 Compiled for: {compiled_arch}")
            
            # Extract major.minor from compute capability
            if '.' in actual_cap:
                major, minor = actual_cap.split('.')
                expected_arch = f"sm_{major}{minor}"
                if expected_arch != compiled_arch:
                    log_and_capture(f"   ❌ ARCHITECTURE MISMATCH!")
                    log_and_capture(f"      Expected: {expected_arch}")
                    log_and_capture(f"      Compiled: {compiled_arch}")
                    log_and_capture(f"      This is likely the root cause of the segmentation fault!")
    except Exception as e:
        log_and_capture(f"   ⚠️  Could not verify GPU architecture: {e}")
    
    try:
        # Check CUDA runtime vs driver compatibility
        runtime_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        driver_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        
        # Extract CUDA versions
        runtime_version = "unknown"
        driver_version = "unknown"
        
        for line in runtime_result.stdout.split('\n'):
            if 'release' in line.lower():
                # Extract version like "release 12.8, V12.8.89"
                import re
                match = re.search(r'release (\d+\.\d+)', line)
                if match:
                    runtime_version = match.group(1)
        
        for line in driver_result.stdout.split('\n'):
            if 'CUDA Version' in line:
                # Extract version like "CUDA Version: 12.8"
                import re
                match = re.search(r'CUDA Version: (\d+\.\d+)', line)
                if match:
                    driver_version = match.group(1)
        
        log_and_capture(f"   🔧 CUDA Runtime: {runtime_version}")
        log_and_capture(f"   🚗 CUDA Driver: {driver_version}")
        
        if runtime_version != "unknown" and driver_version != "unknown":
            if runtime_version > driver_version:
                log_and_capture(f"   ❌ VERSION MISMATCH: Runtime > Driver!")
                log_and_capture(f"      This could cause compatibility issues!")
        
    except Exception as e:
        log_and_capture(f"   ⚠️  Could not check CUDA version compatibility: {e}")
    
    # Check if executable was compiled for correct GPU architecture
    try:
        objdump_result = subprocess.run(["objdump", "-s", "-j", ".nv_fatbin", executable_path], 
                                      capture_output=True, text=True, timeout=10)
        if objdump_result.stdout:
            log_and_capture(f"   🔍 GPU architecture in binary:")
            # Look for architecture markers in the binary
            if "sm_86" in objdump_result.stdout:
                log_and_capture(f"      Found: sm_86 (A10G/RTX 30 series)")
            if "sm_80" in objdump_result.stdout:
                log_and_capture(f"      Found: sm_80 (A100)")
            if "sm_75" in objdump_result.stdout:
                log_and_capture(f"      Found: sm_75 (T4/RTX 20 series)")
            if "sm_70" in objdump_result.stdout:
                log_and_capture(f"      Found: sm_70 (V100)")
    except Exception as e:
        log_and_capture(f"   ⚠️  Could not analyze binary architecture: {e}")

@app.function(
    image=cuda_image,
    gpu="A10G"
)
def list_available_models():
    """List all available models and their configurations."""
    print("📋 Available Models:")
    print("=" * 50)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"🤖 {model_name}")
        print(f"   📊 {config['description']}")
        print(f"   🖥️  Recommended GPU: {config['gpu']}")
        print(f"   💾 Recommended RAM: {config['memory']} MB")
        print(f"   📏 Size: {config['size_gb']} GB")
        if config['s3_key']:
            print(f"   📦 Source: S3 ({config['s3_key']})")
        elif config['download_url']:
            print(f"   🔗 Source: HuggingFace")
        else:
            print(f"   📁 Source: Local file required")
        print()
    
    return True

@app.function(
    image=cuda_image
)
def test_cuda_setup():
    """Test CUDA setup and list available GPUs."""
    print("🔍 Testing CUDA setup...")
    
    try:
        # Test CUDA compiler
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ NVCC Version:\n{result.stdout}")
        
        # Test GPU availability
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print(f"✅ GPU Information:\n{result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ CUDA test failed: {e}")
        return False

@app.function(
    image=cuda_image,
    volumes={"/cache": model_volume},  # Mount the persistent volume
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def manage_volume_cache(action="list"):
    """Manage the persistent volume cache."""
    cache_dir = "/cache"
    
    if action == "list":
        print("📦 Volume Cache Contents:")
        print("=" * 50)
        
        if os.path.exists(cache_dir):
            total_size = 0
            file_count = 0
            
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    total_size += size
                    file_count += 1
                    print(f"📄 {item}: {size / 1e9:.1f} GB")
                else:
                    print(f"📁 {item}/")
            
            print(f"\n📊 Total: {file_count} files, {total_size / 1e9:.1f} GB")
        else:
            print("📂 Volume cache is empty")
    
    elif action == "clear":
        print("🗑️  Clearing volume cache...")
        if os.path.exists(cache_dir):
            import shutil
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"🗑️  Removed: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"🗑️  Removed directory: {item}")
            print("✅ Volume cache cleared")
        else:
            print("📂 Volume cache was already empty")
    
    elif action == "info":
        print("📋 Volume Cache Information:")
        print("=" * 50)
        print(f"📁 Mount path: {cache_dir}")
        print(f"💾 Volume name: huggingface-cache")
        print(f"🔄 Persistent: Yes")
        print(f"📦 Shared across runs: Yes")
        
        if os.path.exists(cache_dir):
            # Get filesystem info
            import shutil
            total, used, free = shutil.disk_usage(cache_dir)
            print(f"💿 Total space: {total / 1e9:.1f} GB")
            print(f"📊 Used space: {used / 1e9:.1f} GB")
            print(f"🆓 Free space: {free / 1e9:.1f} GB")
    
    return True

@app.function(
    image=cuda_image,
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("aws-credentials")]
)
def list_profiling_files():
    """List all profiling files in the cache volume."""
    cache_dir = "/cache"
    
    print("📋 Profiling Files in Cache Volume:")
    print("=" * 50)
    
    if os.path.exists(cache_dir):
        profiling_files = []
        for item in os.listdir(cache_dir):
            is_profile_txt = item.endswith('.txt') and ('profiling' in item.lower() or 'output' in item.lower())
            is_profile_csv = item.endswith('.csv') and ('profile' in item.lower() or 'metrics' in item.lower())
            if is_profile_txt or is_profile_csv:
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    # Get modification time
                    import time
                    mtime = os.path.getmtime(item_path)
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(mtime))
                    profiling_files.append((item, size, formatted_time))
        
        if profiling_files:
            for filename, size, mtime in sorted(profiling_files, key=lambda x: x[2], reverse=True):
                print(f"📄 {filename}")
                print(f"   📏 Size: {size / 1024:.1f} KB")
                print(f"   🕐 Modified: {mtime}")
                print(f"   💾 Download: modal volume get huggingface-cache {filename}")
                print()
        else:
            print("📂 No profiling files found")
    else:
        print("📂 Cache directory not found")
    
    return True

def test_s3_access():
    """Test S3 access and list bucket contents."""
    print("🔍 Testing S3 access...")
    
    if not setup_aws_credentials():
        return False
    
    bucket_name = os.environ.get("S3_BUCKET_NAME", DEFAULT_S3_CONFIG["bucket_name"])
    
    try:
        # Test AWS CLI
        result = subprocess.run(["aws", "s3", "ls", f"s3://{bucket_name}/"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ S3 bucket contents ({bucket_name}):")
        print(result.stdout)
        
        # Check volume contents
        print(f"\n📦 Volume cache contents:")
        cache_dir = "/cache"
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"  📄 {item}: {size / 1e9:.1f} GB")
                else:
                    print(f"  📁 {item}/")
        else:
            print("  📂 Volume cache is empty")
        
        # Test if model exists in S3
        for model_name, config in MODEL_CONFIGS.items():
            if config['s3_key']:
                result = subprocess.run([
                    "aws", "s3", "ls", f"s3://{bucket_name}/{config['s3_key']}"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {model_name} found in S3")
                    
                    # Check if also cached in volume
                    cached_path = f"/cache/{model_name}"
                    if os.path.exists(cached_path):
                        cached_size = os.path.getsize(cached_path)
                        print(f"  📦 Also cached in volume: {cached_size / 1e9:.1f} GB")
                    else:
                        print(f"  📂 Not yet cached in volume")
                else:
                    print(f"❌ {model_name} not found in S3")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ S3 access test failed: {e}")
        print(f"📝 Error output: {e.stderr}")
        return False

@app.local_entrypoint()
def main(
    model: str = "stories15M.bin",
    prompt: str = "Once upon a time",
    steps: int = 256, 
    temperature: float = 1.0,
    topp: float = 0.9,
    seed: int = 42,
    test_only: bool = False,
    list_models: bool = False,
    test_s3: bool = False,
    use_boto3: bool = False,
    cache_action: str = None,  # "list", "clear", "info"
    save_output: bool = True,
    output_file: str = "profiling.txt",
    list_profiling: bool = False,
    cuda_impl: str = "cublas",
    profile_enable: bool = False,
    profile_pos: int = 10,
    profile_csv: str = "/cache/flash_profile_metrics.csv",
    download_dir: str = "."
):
    """
    Main entry point for the Modal CUDA Llama2 application with S3 support.
    
    Args:
        model: Model file to use (stories15M.bin, llama2_7b.bin, or custom)
        prompt: Text prompt for generation
        steps: Number of generation steps  
        temperature: Sampling temperature
        topp: Top-p sampling parameter
        seed: Random seed
        test_only: Only test CUDA setup without running inference
        list_models: List available models and their configurations
        test_s3: Test S3 access and list bucket contents
        use_boto3: Use boto3 instead of AWS CLI for S3 downloads
        cache_action: Volume cache management ("list", "clear", "info")
        save_output: Save output to profiling file in cache volume
        output_file: Name of the output file to save in cache volume
        list_profiling: List saved profiling files in cache volume
        cuda_impl: CUDA implementation source to use ("cublas", "flash", "tiled", or "cublas_fp16")
        profile_enable: Enable per-layer profiling in CUDA binary (not supported by cublas_fp16)
        profile_pos: Token position where profiling is triggered
        profile_csv: CSV output path for stage timings (recommend /cache/...)
    """
    
    print("🌟 Welcome to Modal CUDA Llama2 with S3 Support and Volume Caching!")
    
    if cache_action:
        print(f"🗄️  Managing volume cache: {cache_action}")
        success = manage_volume_cache.remote(action=cache_action)
        if success:
            print(f"✅ Volume cache {cache_action} completed!")
        else:
            print(f"❌ Volume cache {cache_action} failed!")
        return
    
    if list_models:
        print("📋 Listing available models...")
        list_available_models.remote()
        return
    
    if list_profiling:
        print("📋 Listing saved profiling files...")
        list_profiling_files.remote()
        return
    
    if test_s3:
        print("🧪 Testing S3 access...")
        success = test_s3_access.remote()
        if success:
            print("✅ S3 access test passed!")
        else:
            print("❌ S3 access test failed!")
            sys.exit(1)
        return
    
    if test_only:
        print("🧪 Running CUDA setup test only...")
        success = test_cuda_setup.remote()
        if success:
            print("✅ CUDA setup test passed!")
        else:
            print("❌ CUDA setup test failed!")
            sys.exit(1)
        return
    
    print(f"🤖 Model: {model}")
    print(f"📝 Prompt: '{prompt}'")
    print(f"⚙️  Configuration: steps={steps}, temp={temperature}, top-p={topp}, seed={seed}")
    print(f"🧩 CUDA implementation: {cuda_impl}")
    print(f"📈 Profiling: {'enabled' if profile_enable else 'disabled'}")
    print(f"🔧 S3 Method: {'boto3' if use_boto3 else 'AWS CLI'}")
    print(f"💾 Save output: {'Yes' if save_output else 'No'}")
    print(f"📥 Download directory: {download_dir}")

    cuda_impl_key = cuda_impl.strip().lower()
    if cuda_impl_key not in CUDA_SOURCE_VARIANTS:
        print(f"❌ Invalid CUDA implementation: '{cuda_impl}'")
        print(f"💡 Valid options: {', '.join(sorted(CUDA_SOURCE_VARIANTS.keys()))}")
        sys.exit(1)

    # Use implementation-specific default filenames unless user explicitly overrides.
    user_overrode_output_file = (output_file != "profiling.txt")
    user_overrode_profile_csv = (profile_csv != "/cache/flash_profile_metrics.csv")

    if not user_overrode_output_file:
        output_file = f"profiling_{cuda_impl_key}.txt"

    if profile_enable and not user_overrode_profile_csv:
        profile_csv = f"/cache/profiling_{cuda_impl_key}.csv"

    if profile_enable:
        print(f"   trigger position: {profile_pos}")
        print(f"   csv path: {profile_csv}")
    if save_output:
        print(f"📄 Output file: {output_file} (saved to cache volume)")
    
    # Validate model selection
    if model not in MODEL_CONFIGS and not model.endswith('.bin'):
        print(f"⚠️  Warning: Unknown model '{model}'. Supported models:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"   - {model_name}: {config['description']}")
        print("Proceeding with default configuration...")
    
    print("🚀 Starting compilation and inference...")
    
    # Get model configuration for proper resource allocation
    model_config = get_model_config(model)
    print(f"🎯 Using {model_config['gpu']} GPU with {model_config['memory']} MB RAM for {model}")
    
    # Select appropriate function based on model size
    if model_config['size_gb'] > 1.0:  # Large model threshold
        print(f"📦 Using large model function for {model}")
        success = compile_and_run_cuda_large.remote(
            model=model,
            prompt=prompt,
            steps=steps, 
            temperature=temperature,
            topp=topp,
            seed=seed,
            use_boto3=use_boto3,
            save_output=save_output,
            output_file=output_file,
            cuda_impl=cuda_impl_key,
            profile_enable=profile_enable,
            profile_pos=profile_pos,
            profile_csv=profile_csv,
        )
    else:
        print(f"📦 Using small model function for {model}")
        success = compile_and_run_cuda_small.remote(
            model=model,
            prompt=prompt,
            steps=steps, 
            temperature=temperature,
            topp=topp,
            seed=seed,
            use_boto3=use_boto3,
            save_output=save_output,
            output_file=output_file,
            cuda_impl=cuda_impl_key,
            profile_enable=profile_enable,
            profile_pos=profile_pos,
            profile_csv=profile_csv,
        )
    
    if success:
        print(f"🎉 Successfully completed CUDA Llama2 inference with {model}!")
        artifacts_to_download = []
        if save_output:
            print(f"📁 Output saved to cache volume: /cache/{output_file}")
            artifacts_to_download.append(output_file)
        if profile_enable:
            profile_remote_path = profile_csv
            if profile_remote_path.startswith("/cache/"):
                profile_remote_path = profile_remote_path[len("/cache/"):]
            profile_remote_path = profile_remote_path.lstrip("/")
            if profile_remote_path:
                artifacts_to_download.append(profile_remote_path)

        if artifacts_to_download:
            print("📥 Downloading profiling artifacts to local machine...")
            downloaded = set()
            os.makedirs(download_dir, exist_ok=True)
            for remote_path in artifacts_to_download:
                if remote_path in downloaded:
                    continue
                downloaded.add(remote_path)
                local_name = os.path.basename(remote_path)
                local_path = os.path.join(download_dir, local_name)
                remote_candidates = [remote_path]
                if remote_path.startswith("/cache/"):
                    remote_candidates.append(remote_path[len("/cache/"):])
                else:
                    remote_candidates.append(f"/cache/{remote_path}")

                download_ok = False
                for attempt in range(1, 4):
                    for candidate_remote_path in remote_candidates:
                        try:
                            # Use explicit destination filename and avoid capturing output to prevent
                            # Windows console encoding issues from CLI unicode characters.
                            # --force overwrites any existing local file from a previous run.
                            subprocess.run(
                                ["modal", "volume", "get", "--force", "huggingface-cache", candidate_remote_path, local_path],
                                check=True,
                            )
                            print(f"✅ Downloaded: {candidate_remote_path} -> {local_path}")
                            download_ok = True
                            break
                        except subprocess.CalledProcessError as e:
                            last_error = e

                    if download_ok:
                        break

                    print(
                        f"⚠️  Auto-download attempt {attempt}/3 failed for {remote_path}: "
                        f"exit code {last_error.returncode}"
                    )
                    if attempt < 3:
                        # Artifacts can appear with slight delay after remote completion.
                        time.sleep(2)

                if not download_ok:
                    print(f"💡 You can retrieve manually with: modal volume get huggingface-cache {remote_path} {local_path}")
    else:
        print(f"💥 Failed to complete CUDA Llama2 inference with {model}!")
        sys.exit(1)

if __name__ == "__main__":
    # This allows the script to be run locally for development
    print("For local development only. Use 'modal run modal_app.py' for cloud execution.")