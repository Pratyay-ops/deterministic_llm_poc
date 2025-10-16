"""
OFFICIAL Implementation  
Using thinking-machines-lab/batch_invariant_ops with Triton kernels
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

try:
    from batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode, is_batch_invariant_mode_enabled
    BATCH_INVARIANT_AVAILABLE = True
except ImportError:
    BATCH_INVARIANT_AVAILABLE = False


class OfficialImplementation:
    """
    OFFICIAL: thinking-machines-lab/batch_invariant_ops
    
    Uses Triton JIT-compiled kernels for batch invariance
    Expected: Deterministic with all prompts
    
    Requirements:
    - python3-devel (for Triton compilation)
    - batch_invariant_ops repository cloned
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        engine: str = "vllm",  # "vllm" recommended, "hf" also works
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        if not BATCH_INVARIANT_AVAILABLE:
            raise RuntimeError(
                "batch_invariant_ops not available. "
                "Install: git clone https://github.com/thinking-machines-lab/batch_invariant_ops.git"
            )
        
        self.name = "OFFICIAL"
        self.model_name = model_name
        self.device_id = device_id
        self.engine_type = engine
        
        print(f"[OFFICIAL] Initializing with {engine.upper()} engine + Triton kernels...")
        
        # Configure environment BEFORE loading
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        if engine == "vllm":
            from src.engines.vllm_engine import vLLMEngine
            self.engine = vLLMEngine(
                model_name=model_name,
                device_id=device_id,
                dtype=dtype,
                enable_deterministic=True,
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
        
        print(f"[OFFICIAL] ✓ Ready with batch-invariant ops")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs):
        """Generate with batch-invariant ops"""
        with set_batch_invariant_mode(True):
            return self.engine.generate(prompt, max_new_tokens, **kwargs)
    
    def generate_batch(self, prompts: list, max_new_tokens: int = 100, **kwargs):
        """Batch generate with batch-invariant ops"""
        with set_batch_invariant_mode(True):
            return self.engine.generate_batch(prompts, max_new_tokens, **kwargs)
    
    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup()
    
    def get_memory_usage(self):
        """Get memory usage"""
        return self.engine.get_memory_usage()