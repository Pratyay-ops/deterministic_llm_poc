"""
vLLM Inference Engine with Batch-Invariant Operations
Supports deterministic inference using official batch_invariant_ops
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
import torch
from typing import List, Tuple, Dict, Any, Optional
from src.engines.engine_base import DeterministicEngine

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠ vLLM not installed. Install with: pip install vllm")

try:
    from batch_invariant_ops import set_batch_invariant_mode, is_batch_invariant_mode_enabled
    BATCH_INVARIANT_AVAILABLE = True
except ImportError:
    BATCH_INVARIANT_AVAILABLE = False
    print("⚠ batch_invariant_ops not available. Determinism may not work.")


class vLLMEngine(DeterministicEngine):
    """
    vLLM inference engine with batch-invariant operations support
    
    Best for:
    - High throughput serving
    - Continuous batching
    - Production deployments
    - Multi-GPU inference
    
    Determinism:
    - Requires batch_invariant_ops from thinking-machines-lab
    - Requires vLLM with FlexAttention backend
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        dtype: torch.dtype = torch.float16,
        enable_deterministic: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        **kwargs
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not installed. Install with: pip install vllm")
        
        super().__init__(model_name, device_id, dtype)
        
        self.enable_deterministic_mode = enable_deterministic
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
        self.llm = None
        self.deterministic_enabled = False
        
        self.load_model()
        
        if enable_deterministic:
            self.enable_determinism()
    
    def load_model(self) -> None:
        """Load model using vLLM"""
        print(f"[vLLM] Loading {self.model_name}...")
        
        # vLLM configuration
        llm_kwargs = {
            "model": self.model_name,
            "dtype": str(self.dtype).split('.')[-1],  # 'float16' or 'bfloat16'
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": True,  # Disable CUDA graphs for determinism
        }
        
        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len
        
        self.llm = LLM(**llm_kwargs)
        
        print(f"[vLLM] ✓ Model loaded")
    
    def enable_determinism(self) -> None:
        """Enable deterministic inference"""
        if not BATCH_INVARIANT_AVAILABLE:
            print("[vLLM] ⚠ batch_invariant_ops not available, cannot enable determinism")
            return
        
        print("[vLLM] Enabling batch-invariant mode...")
        
        # Set environment variables
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Configure PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        self.deterministic_enabled = True
        print("[vLLM] ✓ Determinism enabled")
    
    def disable_determinism(self) -> None:
        """Disable deterministic inference"""
        self.deterministic_enabled = False
        print("[vLLM] Determinism disabled")
    
    def verify_determinism(self) -> bool:
        """Verify determinism is properly configured"""
        if not BATCH_INVARIANT_AVAILABLE:
            return False
        
        checks = [
            torch.backends.cudnn.deterministic,
            not torch.backends.cudnn.benchmark,
            not torch.backends.cuda.matmul.allow_tf32,
        ]
        
        return all(checks)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[str, float]:
        """Generate text using vLLM"""
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1.0 if temperature == 0.0 else 0.9,
            top_k=-1 if temperature == 0.0 else 50,
        )
        
        start_time = time.time()
        
        if self.deterministic_enabled and BATCH_INVARIANT_AVAILABLE:
            with set_batch_invariant_mode(True):
                outputs = self.llm.generate([prompt], sampling_params)
        else:
            outputs = self.llm.generate([prompt], sampling_params)
        
        elapsed = time.time() - start_time
        
        generated_text = outputs[0].outputs[0].text
        
        return generated_text, elapsed
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Generate text for multiple prompts using vLLM's batching"""
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1.0 if temperature == 0.0 else 0.9,
            top_k=-1 if temperature == 0.0 else 50,
        )
        
        start_time = time.time()
        
        if self.deterministic_enabled and BATCH_INVARIANT_AVAILABLE:
            with set_batch_invariant_mode(True):
                outputs = self.llm.generate(prompts, sampling_params)
        else:
            outputs = self.llm.generate(prompts, sampling_params)
        
        total_elapsed = time.time() - start_time
        per_prompt_time = total_elapsed / len(prompts)
        
        results = [
            (output.outputs[0].text, per_prompt_time)
            for output in outputs
        ]
        
        return results
    
    def cleanup(self) -> None:
        """Clean up vLLM resources"""
        if self.llm is not None:
            del self.llm
        self.clear_cache()