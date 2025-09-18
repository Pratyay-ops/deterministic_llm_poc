# deterministic_mps_inference.py
"""
Deterministic LLM inference using M4 GPU cores via MPS
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.batch_invariant_ops_mps import set_mps_batch_invariant_mode
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
            torch_dtype=torch.float16,  # Use float16 for GPU efficiency
            trust_remote_code=True,
            device_map={"": self.device},  # Force everything to MPS
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        # Monkey-patch attention mechanism for batch invariance
        self._patch_attention_for_determinism()
        
        # Generation config
        self.generation_config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=1,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def _patch_attention_for_determinism(self):
        """
        Patch attention layers to ensure batch invariance
        Following thinking-machines-lab approach
        """
        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                # Store original forward
                original_forward = module.forward
                
                def deterministic_forward(self, *args, **kwargs):
                    # Force attention computation to use fixed patterns
                    with set_mps_batch_invariant_mode(True):
                        return original_forward(*args, **kwargs)
                
                # Replace forward method
                module.forward = deterministic_forward.__get__(module, module.__class__)
    
    def generate(self, prompt: str, max_new_tokens: int = 100) -> tuple[str, float]:
        """Generate with timing information"""
        
        start_time = time.time()
        
        with set_mps_batch_invariant_mode(True):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Use torch.mps.synchronize() for accurate timing
                torch.mps.synchronize()
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens
                )
                
                torch.mps.synchronize()
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        elapsed = time.time() - start_time
        
        return response, elapsed