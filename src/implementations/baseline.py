"""
BASELINE Implementation
Standard inference without any determinism modifications
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
from src.engines.hf_engine import HuggingFaceEngine


class BaselineImplementation:
    """
    BASELINE: Standard PyTorch/Transformers inference
    
    No modifications for determinism
    Expected: Non-deterministic with complex prompts
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        engine: str = "hf",  # "hf" or "vllm"
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        self.name = "BASELINE"
        self.model_name = model_name
        self.device_id = device_id
        self.engine_type = engine
        
        print(f"[BASELINE] Initializing with {engine.upper()} engine...")
        
        if engine == "vllm":
            from src.engines.vllm_engine import vLLMEngine
            self.engine = vLLMEngine(
                model_name=model_name,
                device_id=device_id,
                dtype=dtype,
                enable_deterministic=False,  # BASELINE doesn't use determinism
                **kwargs
            )
        else:  # Default to HuggingFace
            self.engine = HuggingFaceEngine(
                model_name=model_name,
                device_id=device_id,
                dtype=dtype,
                **kwargs
            )
        
        print(f"[BASELINE] ✓ Ready")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs):
        """Generate without determinism"""
        return self.engine.generate(prompt, max_new_tokens, **kwargs)
    
    def generate_batch(self, prompts: list, max_new_tokens: int = 100, **kwargs):
        """Batch generate without determinism"""
        return self.engine.generate_batch(prompts, max_new_tokens, **kwargs)
    
    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup()
    
    def get_memory_usage(self):
        """Get memory usage"""
        return self.engine.get_memory_usage()