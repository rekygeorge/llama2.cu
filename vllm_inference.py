import modal
import os
from typing import Optional

# Create Modal app
app = modal.App("llama2-story-generator")

# OPTION 1: Set your HF token directly here (not recommended for production)
HF_TOKEN =  os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

print("HF_TOKEN : ", HF_TOKEN , "<---")

# OPTION 2: Or read from environment variable
# HF_TOKEN = os.getenv("HF_TOKEN", "hf_your_token_here")

# Create Modal secret for Hugging Face token
# You need to set this up in Modal dashboard: modal secret create huggingface-secret HF_TOKEN=your_token_here

# Define the container image with vLLM and dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "numpy<2.0",          # Pin NumPy to 1.x for compatibility
        "fastapi[standard]",  # Required for web endpoints - install first
        "vllm==0.2.7",
        "transformers>=4.30.0", 
        "torch>=2.0.0",
        "accelerate",
    ])
    .run_commands("echo 'FastAPI added with NumPy 1.x - v3'")  # Force rebuild
)

# Configure GPU resources - A100 with sufficient memory
@app.cls(
    image=image,
    gpu="A100-40GB",  # Updated GPU specification
    memory=32768,  # 32GB RAM
    timeout=600,   # 10 minutes timeout
    scaledown_window=300,  # Updated parameter name
    # No secrets needed when passing token directly
)
class LlamaStoryGenerator:
    @modal.enter()
    def setup(self):
        from vllm import LLM, SamplingParams
        import os
        
        # Use the token defined at module level
        if not HF_TOKEN or HF_TOKEN == "hf_your_token_here":
            raise ValueError("Please set your HF_TOKEN in the code. Get it from https://huggingface.co/settings/tokens")
        
        # Set HF token for transformers library
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        os.environ["HF_TOKEN"] = HF_TOKEN
        
        print(f"Using HF token: {HF_TOKEN[:10]}...")  # Show first 10 chars for verification
        
        # Initialize the vLLM engine with Llama2-7B
        self.llm = LLM(
            model="meta-llama/Llama-2-7b-chat-hf",         # earlier it was "meta-llama/Llama-2-7b-chat-hf"
            tensor_parallel_size=1,  # Single A100
            dtype="float32",         # Use FP16 for memory efficiency , earlier it was float16
            trust_remote_code=True,
            max_model_len=2048 ,      # Context length, earlier it was 4096
            gpu_memory_utilization=0.95,  # Use 95% of GPU memory
            download_dir="/tmp/model_cache",  # Cache directory
            revision="main",
        )
        
        # Configure sampling parameters for story generation
        self.sampling_params = SamplingParams(
            temperature=0.8,      # Creative but not too random
            top_p=0.9,           # Nucleus sampling
            max_tokens=512,      # Story length
            repetition_penalty=1.1,
            stop=["</s>", "<|endoftext|>"]  # Stop tokens
        )
    
    @modal.method()
    def generate_story(self, 
                      prompt: str = "Once upon a time",
                      max_tokens: int = 512,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      top_k: int = -1,
                      repetition_penalty: float = 1.1,
                      seed: int = None,
                      stop_sequences: list = None) -> dict:
        """Generate a story with detailed metrics and configurable parameters"""
        import time
        from vllm import SamplingParams
        
        # Create custom sampling params for this request
        if stop_sequences is None:
            stop_sequences = ["</s>", "<|endoftext|>"]
        
        # Build sampling params dict, excluding unsupported parameters
        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stop": stop_sequences,
        }
        
        # Only add top_k if it's not -1 (disabled)
        if top_k > 0:
            sampling_kwargs["top_k"] = top_k
            
        # Note: vLLM 0.2.7 doesn't support seed parameter in SamplingParams
        # Seed control would need to be set at engine level or via environment
        if seed is not None:
            print(f"⚠️  Note: Seed {seed} specified but not supported in vLLM 0.2.7 SamplingParams")
            
        sampling_params = SamplingParams(**sampling_kwargs)
        
        # Format prompt for Llama2-Chat
        formatted_prompt = f"""<s>[INST] Write a creative and engaging story that begins with: "{prompt}"

Please create an imaginative story with interesting characters and plot. Make it entertaining and suitable for all ages. [/INST]"""
        
        print(f"🚀 Starting generation with parameters:")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
        print(f"   Top-k: {top_k if top_k > 0 else 'disabled'}")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Seed: {seed if seed is not None else 'not set'}")
        print(f"   Repetition penalty: {repetition_penalty}")
        
        # Time the generation
        start_time = time.time()
        
        # Generate the story
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Extract output and metrics
        output = outputs[0]
        generated_text = output.outputs[0].text.strip()
        
        # Calculate metrics
        prompt_tokens = len(output.prompt_token_ids)
        generated_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + generated_tokens
        
        tokens_per_second = generated_tokens / generation_time if generation_time > 0 else 0
        
        # Print detailed metrics
        print(f"\n📊 Generation Metrics:")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Generated tokens: {generated_tokens}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Tokens/second: {tokens_per_second:.2f}")
        print(f"   Finish reason: {output.outputs[0].finish_reason}")
        
        return {
            "generated_text": generated_text,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": total_tokens,
                "generation_time_seconds": generation_time,
                "tokens_per_second": tokens_per_second,
                "finish_reason": output.outputs[0].finish_reason,
            },
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "seed": seed,
                "repetition_penalty": repetition_penalty,
            }
        }

# Function to run inference from client
@app.function()
def run_story_generation(prompt: str = "Once upon a time",
                        max_tokens: int = 512,
                        temperature: float = 0.8,
                        top_p: float = 0.9,
                        top_k: int = -1,
                        repetition_penalty: float = 1.1,
                        seed: int = None):
    """Client function to generate a story with full parameter control"""
    generator = LlamaStoryGenerator()
    result = generator.generate_story.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed
    )
    return result

# Local function to run the Modal app
@app.local_entrypoint()
def main(prompt: str = "Once upon a time",
         max_tokens: int = 512,
         temperature: float = 0.8,
         top_p: float = 0.9,
         top_k: int = -1,
         repetition_penalty: float = 1.1,
         seed: int = None):
    """Main entry point for local execution with full parameter control"""
    print(f"🎯 Generating story with prompt: '{prompt}'")
    print("=" * 60)
    
    result = run_story_generation.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed
    )
    
    print("\n📖 Generated Story:")
    print("=" * 60)
    print(result["generated_text"])
    print("=" * 60)
    
    print(f"\n🔢 Final Metrics Summary:")
    metrics = result["metrics"]
    print(f"   • Generated {metrics['generated_tokens']} tokens in {metrics['generation_time_seconds']:.2f}s")
    print(f"   • Speed: {metrics['tokens_per_second']:.1f} tokens/second")
    print(f"   • Total tokens processed: {metrics['total_tokens']}")
    
    return result

# Alternative: Web endpoint for API access (commented out to avoid FastAPI requirement)
# @app.function()
# @modal.web_endpoint(method="POST")
# def generate_story_api(item: dict):
#     """Web API endpoint for story generation"""
#     prompt = item.get("prompt", "Once upon a time")
#     max_length = item.get("max_tokens", 512)
#     temperature = item.get("temperature", 0.8)
#     
#     # Create generator instance and generate story
#     generator = LlamaStoryGenerator()
#     story = generator.generate_story.remote(prompt)
#     
#     return {
#         "prompt": prompt,
#         "story": story,
#         "model": "meta-llama/Llama-2-7b-chat-hf",
#         "parameters": {
#             "max_tokens": max_length,
#             "temperature": temperature
#         }
#     }

# Example usage function with different parameter combinations
def example_usage():
    """Example of how to use the story generator with different parameters"""
    
    # Different parameter combinations to try
    test_cases = [
        {
            "prompt": "Once upon a time",
            "temperature": 0.3,  # More focused/deterministic
            "max_tokens": 200,
            "seed": 42
        },
        {
            "prompt": "Once upon a time in a magical forest",
            "temperature": 1.0,  # Very creative
            "top_p": 0.95,
            "max_tokens": 300,
            "repetition_penalty": 1.2
        },
        {
            "prompt": "Once upon a time there was a brave robot",
            "temperature": 0.8,
            "top_k": 50,  # Limit to top 50 tokens
            "max_tokens": 400,
            "seed": 123
        }
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"🧪 TEST CASE {i}: {params['prompt'][:40]}...")
        print(f"{'='*80}")
        
        result = run_story_generation.remote(**params)
        
        print(f"📖 Story:\n{result['generated_text']}")
        print(f"\n📊 Metrics: {result['metrics']['tokens_per_second']:.1f} tok/s, "
              f"{result['metrics']['generated_tokens']} tokens")

# Advanced parameter exploration function
@app.function()
def compare_parameters():
    """Compare different parameter settings side by side"""
    
    base_prompt = "Once upon a time in a mysterious castle"
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 0.8, 1.2]
    
    generator = LlamaStoryGenerator()
    
    print("🔬 TEMPERATURE COMPARISON:")
    print("=" * 80)
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        print("-" * 40)
        
        result = generator.generate_story.remote(
            prompt=base_prompt,
            temperature=temp,
            max_tokens=150,
            seed=42  # Same seed for fair comparison
        )
        
        print(f"Story: {result['generated_text'][:200]}...")
        print(f"Speed: {result['metrics']['tokens_per_second']:.1f} tok/s")
    
    return "Parameter comparison complete!"

if __name__ == "__main__":
    # Run with default prompt
    main()

# Setup instructions:
"""
SETUP INSTRUCTIONS:

1. Get Hugging Face Token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with 'Read' permissions
   - Accept Meta's Llama2 license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

2. Update the code:
   - Replace HF_TOKEN = "hf_your_token_here" with your actual token
   - Or set environment variable: export HF_TOKEN=your_hf_token_here

3. Install Modal:
   pip install modal

4. Setup Modal account:
   modal setup

5. Run the code:
   modal run llama2_modal_inference.py

SECURITY NOTE: 
- Hardcoding tokens in code is not recommended for production
- Consider using environment variables or Modal secrets for production use
- Don't commit your token to version control

Alternative methods to pass token:
- Environment variable: HF_TOKEN = os.getenv("HF_TOKEN", "fallback_token")
- Command line: export HF_TOKEN=your_token && modal run script.py
- Modal secrets (original method): more secure for production
"""