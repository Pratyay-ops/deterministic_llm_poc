"""Validation utilities for determinism testing"""

import torch
from typing import List, Tuple
import hashlib
from .batch_invariant_ops_cuda import set_mps_batch_invariant_mode

def validate_batch_invariance() -> bool:
    """
    Test batch invariance following thinking-machines-lab approach
    """
    B, D = 128, 256
    
    # Create test tensors
    a = torch.linspace(-10, 10, B*D).reshape(B, D).to('mps')
    b = torch.linspace(-10, 10, D*D).reshape(D, D).to('mps')
    
    with set_mps_batch_invariant_mode(True):
        # Single sample
        out1 = torch.mm(a[:1], b)
        
        # Full batch then slice
        out2 = torch.mm(a, b)[:1]
        
        # Check difference
        diff = (out1 - out2).abs().max().item()
        
    return diff < 1e-6

def validate_model_outputs(
    model,
    prompt: str,
    num_runs: int = 5
) -> Tuple[bool, List[str]]:
    """
    Validate that model produces identical outputs
    """
    outputs = []
    hashes = []
    
    for _ in range(num_runs):
        output, _ = model.generate(prompt, max_new_tokens=50)
        outputs.append(output)
        hashes.append(hashlib.sha256(output.encode()).hexdigest())
    
    is_deterministic = len(set(hashes)) == 1
    return is_deterministic, outputs