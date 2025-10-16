"""
CUSTOM Implementation
Our CUDA batch-invariant operations with inference engine support
"""

import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional
from src.utils.cuda_batch_invariant_ops import set_cuda_batch_invariant_mode


class CustomImplementation:
    """
    CUSTOM: Our CUDA batch-invariant implementation
    
    Uses:
    - Fixed-chunk matrix multiplication (32-element chunks)
    - Kahan summation for reductions
    - CUDA determinism configuration
    
    Expected: Deterministic with all prompts
    Advantage: No Triton dependency, works with any engine
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        engine: str = "vllm",  # "vllm" or "hf"
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        self.name = "CUSTOM"
        self.model_name = model_name
        self.device_id = device_id
        self.engine_type = engine
        
        print(f"[CUSTOM] Initializing with {engine.upper()} engine + CUDA batch-invariant ops...")
        
        if engine == "vllm":
            from src.engines.vllm_engine import vLLMEngine
            self.engine = vLLMEngine(
                model_name=model_name,
                device_id=device_id,
                dtype=dtype,
                enable_deterministic=False,  # We handle determinism via context manager
                **kwargs
            )
        else:
            from src.engines.hf_engine import HuggingFaceEngine
            self.engine = HuggingFaceEngine(
                model_name=model_name,
                device_id=device_id,
                dtype=dtype,
                **kwargs
            )
        
        print(f"[CUSTOM] ✓ Ready")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs):
        """Generate with CUDA batch-invariant ops"""
        with set_cuda_batch_invariant_mode(True):
            return self.engine.generate(prompt, max_new_tokens, **kwargs)
    
    def generate_batch(self, prompts: list, max_new_tokens: int = 100, **kwargs):
        """Batch generate with CUDA batch-invariant ops"""
        with set_cuda_batch_invariant_mode(True):
            return self.engine.generate_batch(prompts, max_new_tokens, **kwargs)
    
    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup()
    
    def get_memory_usage(self):
        """Get memory usage"""
        return self.engine.get_memory_usage()