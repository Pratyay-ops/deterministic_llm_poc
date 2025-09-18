"""Utility functions for M4 optimization"""

import torch
import psutil
import hashlib
from typing import Dict, Any

def check_mps_availability() -> bool:
    """Check if MPS is available and working"""
    if not torch.backends.mps.is_available():
        return False
    
    try:
        # Test MPS with a simple operation
        test_tensor = torch.tensor([1.0]).to('mps')
        _ = test_tensor * 2
        return True
    except:
        return False

def get_m4_config() -> Dict[str, Any]:
    """Get optimal configuration for M4 based on available RAM"""
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if ram_gb >= 24:
        return {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dtype": torch.float16,
            "max_batch_size": 4,
            "max_sequence_length": 4096,
            "device": "mps"
        }
    else:
        return {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "dtype": torch.float16,
            "max_batch_size": 2,
            "max_sequence_length": 2048,
            "device": "mps"
        }

def hash_output(text: str) -> str:
    """Generate hash of text for determinism checking"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def measure_memory_usage():
    """Measure current memory usage"""
    if torch.backends.mps.is_available():
        # MPS memory
        allocated = torch.mps.current_allocated_memory() / 1024**3
        reserved = torch.mps.driver_allocated_memory() / 1024**3
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved
        }
    return {"allocated_gb": 0, "reserved_gb": 0}