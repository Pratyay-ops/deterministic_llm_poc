# deterministic_cuda_inference.py
"""
Deterministic LLM inference using NVIDIA GPUs via CUDA
Optimized for NVIDIA A10G GPUs with 24GB VRAM each
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from batch_invariant_ops_cuda import set_cuda_batch_invariant_mode, verify_cuda_determinism
import time
from typing import Optional, Dict, Any


class DeterministicCUDAModel:
    """
    Deterministic LLM inference on NVIDIA GPUs
    
    This implementation ensures 100% deterministic outputs by:
    1. Using batch-invariant operations (consistent reduction order)
    2. Disabling non-deterministic CUDA optimizations
    3. Using fixed random seeds
    4. Processing with consistent chunk sizes
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device_id: int = 0,
        torch_dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = False
    ):
        """
        Initialize deterministic model on CUDA
        
        Args:
            model_name: HuggingFace model identifier
            device_id: CUDA device ID (0-3 for g5.12xlarge with 4 GPUs)
            torch_dtype: Data type for model weights (float16 recommended for A10G)
            use_flash_attention: Whether to use flash attention (if available)
        """
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This requires NVIDIA GPUs.")
        
        # Verify determinism configuration
        print("Verifying CUDA determinism configuration...")
        if not verify_cuda_determinism():
            print("⚠ Warning: Some determinism checks failed")
        
        self.device = torch.device(f"cuda:{device_id}")
        self.device_id = device_id
        
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"Device: {torch.cuda.get_device_name(device_id)} (cuda:{device_id})")
        print(f"VRAM: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")
        print(f"Dtype: {torch_dtype}")
        print(f"{'='*60}\n")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading configuration for A10G GPUs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": {"": self.device},  # Force all layers to specific device
            "low_cpu_mem_usage": True,
        }
        
        # Add flash attention if requested (requires flash-attn package)
        if use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using Flash Attention 2")
            except:
                print("Flash Attention not available, using standard attention")
        
        # Load model
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        load_time = time.time() - start_time
        
        self.model.eval()
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Patch attention mechanism for batch invariance
        self._patch_attention_for_determinism()
        
        # Generation config for deterministic output
        self.generation_config = GenerationConfig(
            temperature=0.0,  # Greedy decoding
            do_sample=False,  # No sampling
            top_p=1.0,
            top_k=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # Enable KV cache for efficiency
        )
        
        print("✓ Model configured for deterministic inference\n")
    
    def _patch_attention_for_determinism(self):
        """
        Patch attention layers to ensure batch invariance
        
        This ensures that attention computations follow the same reduction order
        regardless of batch size, preventing numerical differences due to
        floating-point non-associativity.
        """
        attention_layers_patched = 0
        
        for name, module in self.model.named_modules():
            # Identify attention layers (works for most transformer architectures)
            if any(keyword in name.lower() for keyword in ["attention", "attn", "self_attn"]):
                # Store original forward method
                original_forward = module.forward
                
                def create_deterministic_forward(original_fn):
                    """Closure to capture the original forward function"""
                    def deterministic_forward(*args, **kwargs):
                        # Force attention computation to use batch-invariant operations
                        with set_cuda_batch_invariant_mode(True):
                            return original_fn(*args, **kwargs)
                    return deterministic_forward
                
                # Replace forward method with deterministic version
                module.forward = create_deterministic_forward(original_forward)
                attention_layers_patched += 1
        
        print(f"✓ Patched {attention_layers_patched} attention layers for determinism")
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        return_timing: bool = True
    ) -> tuple[str, Optional[float]]:
        """
        Generate deterministic output for a given prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            return_timing: Whether to return generation time
        
        Returns:
            Tuple of (generated_text, generation_time)
        """
        
        start_time = time.time()
        
        # Use batch-invariant mode for entire generation process
        with set_cuda_batch_invariant_mode(True):
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                # Synchronize CUDA before timing
                torch.cuda.synchronize(self.device)
                gen_start = time.time()
                
                # Generate with deterministic settings
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens
                )
                
                # Synchronize CUDA after generation
                torch.cuda.synchronize(self.device)
                gen_time = time.time() - gen_start
            
            # Decode output (only the newly generated tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        total_time = time.time() - start_time
        
        if return_timing:
            return response, total_time
        return response, None
    
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        batch_size: int = 4
    ) -> list[tuple[str, float]]:
        """
        Generate deterministic outputs for multiple prompts
        
        Processes prompts in batches for efficiency while maintaining determinism.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            batch_size: Number of prompts to process simultaneously
        
        Returns:
            List of (generated_text, generation_time) tuples
        """
        results = []
        
        with set_cuda_batch_invariant_mode(True):
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                
                with torch.no_grad():
                    start_time = time.time()
                    torch.cuda.synchronize(self.device)
                    
                    # Generate
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        max_new_tokens=max_new_tokens
                    )
                    
                    torch.cuda.synchronize(self.device)
                    elapsed = time.time() - start_time
                
                # Decode outputs
                for j, output in enumerate(outputs):
                    response = self.tokenizer.decode(
                        output[inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    results.append((response, elapsed / len(batch_prompts)))
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
        }
    
    def clear_cache(self):
        """Clear CUDA cache"""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)


def demonstrate_determinism(model: DeterministicCUDAModel, num_runs: int = 5):
    """
    Demonstrate deterministic behavior by generating the same output multiple times
    
    Args:
        model: DeterministicCUDAModel instance
        num_runs: Number of times to generate output
    """
    prompt = "Explain quantum computing in simple terms:"
    
    print(f"\n{'='*60}")
    print(f"DETERMINISM DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Runs: {num_runs}")
    print(f"{'='*60}\n")
    
    outputs = []
    times = []
    
    for i in range(num_runs):
        output, elapsed = model.generate(prompt, max_new_tokens=50)
        outputs.append(output)
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")
    
    # Check determinism
    unique_outputs = len(set(outputs))
    
    print(f"\n{'='*60}")
    if unique_outputs == 1:
        print("✓ 100% DETERMINISTIC - All outputs identical!")
    else:
        print(f"✗ NON-DETERMINISTIC - {unique_outputs} unique outputs")
    print(f"Average time: {sum(times)/len(times):.3f}s")
    print(f"{'='*60}\n")
    
    print("Sample output:")
    print(outputs[0][:200] + ("..." if len(outputs[0]) > 200 else ""))
    print()
    
    return unique_outputs == 1