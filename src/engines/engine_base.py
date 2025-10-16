"""
Abstract base class for inference engines
All engines must implement this interface for consistent testing
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch


class InferenceEngine(ABC):
    """
    Abstract base class for LLM inference engines
    
    All implementations (vLLM, HuggingFace, SGLang) must inherit from this
    to ensure consistent interface for testing.
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        self.model_name = model_name
        self.device_id = device_id
        self.dtype = dtype
        self.device = torch.device(f"cuda:{device_id}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Tuple[str, float]:
        """
        Generate text for a single prompt
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Engine-specific parameters
        
        Returns:
            Tuple of (generated_text, generation_time_seconds)
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Generate text for multiple prompts
        
        Args:
            prompts: List of input texts
            max_new_tokens: Maximum tokens to generate
            **kwargs: Engine-specific parameters
        
        Returns:
            List of (generated_text, generation_time_seconds) tuples
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (memory, cache, etc.)"""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            }
        return {}
    
    def clear_cache(self) -> None:
        """Clear CUDA cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)


class DeterministicEngine(InferenceEngine):
    """
    Extended base class for deterministic inference engines
    Adds determinism-specific methods
    """
    
    @abstractmethod
    def enable_determinism(self) -> None:
        """Enable deterministic mode"""
        pass
    
    @abstractmethod
    def disable_determinism(self) -> None:
        """Disable deterministic mode"""
        pass
    
    @abstractmethod
    def verify_determinism(self) -> bool:
        """Verify determinism configuration"""
        pass
