# src/deterministic_mps_inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from .batch_invariant_ops_mps import set_mps_batch_invariant_mode
import time

class DeterministicMPSModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize model on MPS (M4 GPU)"""
        
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available. This requires Apple Silicon.")
        
        self.device = torch.device("mps")
        print(f"Loading {model_name} on M4 GPU via MPS...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model optimized for MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,  # Fixed from torch_dtype
            trust_remote_code=True,
            device_map={"": self.device},
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Generation config
        self.generation_config = GenerationConfig(
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> tuple[str, float]:
        """Generate with timing information"""
        
        start_time = time.time()
        
        # Don't nest context managers - it's handled in benchmark.py
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            # Use torch.mps.synchronize() for accurate timing
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens
            )
            
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        elapsed = time.time() - start_time
        
        return response, elapsed