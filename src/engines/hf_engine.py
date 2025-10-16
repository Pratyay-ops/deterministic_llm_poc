"""
HuggingFace Transformers Inference Engine
Standard baseline implementation without optimizations
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
import torch
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.engines.engine_base import InferenceEngine


class HuggingFaceEngine(InferenceEngine):
    """
    Standard HuggingFace Transformers inference
    
    Best for:
    - Baseline comparison
    - Simple deployment
    - Maximum compatibility
    - Development/testing
    
    Not optimized for:
    - High throughput
    - Continuous batching
    """
    
    def __init__(
        self,
        model_name: str,
        device_id: int = 0,
        dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = False,
        **kwargs
    ):
        super().__init__(model_name, device_id, dtype)
        
        self.use_flash_attention = use_flash_attention
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load model using HuggingFace Transformers"""
        print(f"[HF] Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "device_map": {"": self.device},
            "low_cpu_mem_usage": True,
        }
        
        if self.use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("[HF] Using Flash Attention 2")
            except:
                print("[HF] Flash Attention not available")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()
        
        # Generation config
        self.generation_config = GenerationConfig(
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        
        print(f"[HF] ✓ Model loaded on {self.device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Tuple[str, float]:
        """Generate text using HuggingFace"""
        
        start_time = time.time()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)
        
        with torch.no_grad():
            torch.cuda.synchronize(self.device)
            
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens
            )
            
            torch.cuda.synchronize(self.device)
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        elapsed = time.time() - start_time
        
        return generated_text, elapsed
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        batch_size: int = 4,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Generate text for multiple prompts with batching"""
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            start_time = time.time()
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)
            
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens
                )
                
                torch.cuda.synchronize(self.device)
            
            elapsed = time.time() - start_time
            per_prompt_time = elapsed / len(batch_prompts)
            
            for j, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(
                    output[inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                results.append((generated_text, per_prompt_time))
        
        return results
    
    def cleanup(self) -> None:
        """Clean up HuggingFace resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.clear_cache()