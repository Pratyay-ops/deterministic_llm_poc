# src/validation.py

import torch
from typing import List, Tuple
import hashlib

def validate_batch_invariance() -> bool:
    """
    Test batch invariance following thinking-machines-lab approach
    """
    try:
        from .batch_invariant_ops_mps import set_mps_batch_invariant_mode
    except:
        from batch_invariant_ops_mps import set_mps_batch_invariant_mode
    
    if not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = 'cpu'
    else:
        device = 'mps'
    
    B, D = 128, 256
    
    # Create test tensors
    a = torch.linspace(-10, 10, B*D).reshape(B, D).to(device)
    b = torch.linspace(-10, 10, D*D).reshape(D, D).to(device)
    
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
    
    for i in range(num_runs):
        try:
            # Handle different return types from generate method
            result = model.generate(prompt, max_new_tokens=50)
            
            # Check if it returns tuple or just string
            if isinstance(result, tuple):
                output = result[0]
            else:
                output = result
                
            outputs.append(output)
            hashes.append(hashlib.sha256(output.encode()).hexdigest())
        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            outputs.append("")
            hashes.append("")
    
    # Check if all non-empty outputs are identical
    valid_hashes = [h for h in hashes if h]
    is_deterministic = len(set(valid_hashes)) == 1 if valid_hashes else False
    
    return is_deterministic, outputs