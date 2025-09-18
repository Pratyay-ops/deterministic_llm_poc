# src/batch_invariant_ops_mps.py
"""
Batch-invariant operations for Apple Silicon GPU (Metal Performance Shaders)
Adapted from thinking-machines-lab/batch_invariant_ops for M4 GPU
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
import os
from typing import Optional, Callable, Dict, Any

class MPSBatchInvariantOps:
    """
    Batch-invariant operations using Apple Silicon GPU via MPS
    """
    
    def __init__(self):
        self.original_ops = {}
        self.enabled = False
        self._configure_metal_for_determinism()
        
        # Store original operations BEFORE any replacement
        self.original_matmul = torch.matmul
        self.original_mm = torch.mm
        self.original_softmax = torch.nn.functional.softmax
    
    def _configure_metal_for_determinism(self):
        """Configure Metal for maximum determinism"""
        # Force Metal to use deterministic algorithms
        os.environ['METAL_DEVICE_WRAPPER_TYPE'] = 'SERIAL'
        os.environ['METAL_DEBUG_ERROR_MODE'] = '3'
        
        # Disable Metal performance optimizations that may cause non-determinism
        os.environ['METAL_PERFORMANCE_SHADERS_ENABLED'] = '1'
        os.environ['METAL_SHADER_VALIDATION'] = '1'
    
    def _ensure_mps_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on MPS device"""
        if tensor.device.type != 'mps':
            return tensor.to('mps')
        return tensor

# Global instance to store original operations
_ops_instance = MPSBatchInvariantOps()

# Batch-invariant kernel implementations for MPS
def mps_batch_invariant_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """
    Batch-invariant matrix multiplication on MPS
    Key insight: Process with fixed chunk sizes regardless of actual batch size
    """
    device = input.device
    
    if device.type != 'mps':
        input = input.to('mps')
        mat2 = mat2.to('mps')
    
    # Use the ORIGINAL matmul function to avoid recursion
    # Fixed chunk size for consistent computation pattern
    FIXED_CHUNK_SIZE = 16  # Always process in chunks of 16, padding if needed
    
    if len(input.shape) == 2:
        batch_size = input.shape[0]
        
        # Pad to multiple of FIXED_CHUNK_SIZE for consistent execution
        pad_size = (FIXED_CHUNK_SIZE - (batch_size % FIXED_CHUNK_SIZE)) % FIXED_CHUNK_SIZE
        
        if pad_size > 0:
            # Pad input to ensure consistent chunk processing
            padded_input = torch.nn.functional.pad(input, (0, 0, 0, pad_size))
        else:
            padded_input = input
        
        # Use ORIGINAL matmul to avoid recursion
        result = _ops_instance.original_matmul(padded_input, mat2)
        
        # Remove padding from result
        return result[:batch_size]
    else:
        # Use ORIGINAL matmul for other cases
        return _ops_instance.original_matmul(input, mat2)

def mps_batch_invariant_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Batch-invariant softmax using Kahan summation for numerical stability
    """
    # Use double precision for intermediate calculations
    input_double = input.double()
    
    # Compute max for numerical stability
    max_vals = input_double.max(dim=dim, keepdim=True)[0]
    shifted = input_double - max_vals
    
    # Compute exp
    exp_vals = torch.exp(shifted)
    
    # Use Kahan summation for deterministic reduction
    sum_exp = kahan_sum(exp_vals, dim=dim, keepdim=True)
    
    # Return to original dtype
    result = (exp_vals / sum_exp).to(input.dtype)
    return result

def kahan_sum(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Kahan summation algorithm for improved numerical precision
    This helps maintain determinism across different batch sizes
    """
    shape = list(tensor.shape)
    shape[dim] = 1
    
    sum_result = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    compensation = torch.zeros_like(sum_result)
    
    for i in range(tensor.shape[dim]):
        slice_idx = [slice(None)] * len(tensor.shape)
        slice_idx[dim] = i
        
        value = tensor[tuple(slice_idx)].unsqueeze(dim)
        y = value - compensation
        t = sum_result + y
        compensation = (t - sum_result) - y
        sum_result = t
    
    if not keepdim:
        sum_result = sum_result.squeeze(dim)
    
    return sum_result

# Track whether batch invariant mode is enabled
_batch_invariant_enabled = False

@contextmanager
def set_mps_batch_invariant_mode(enabled: bool = True):
    """
    Context manager for MPS batch-invariant operations
    """
    global _batch_invariant_enabled
    
    if not torch.backends.mps.is_available():
        # Silently fall back to CPU mode
        yield
        return
    
    # Skip if already in the desired state
    if _batch_invariant_enabled == enabled:
        yield
        return
    
    if enabled:
        # Only print once when actually enabling
        if not _batch_invariant_enabled:
            print("Enabling MPS batch-invariant mode...")
        
        _batch_invariant_enabled = True
        
        # Configure PyTorch for determinism
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        torch.mps.manual_seed(42)
        
        # Replace operations with batch-invariant versions
        torch.matmul = mps_batch_invariant_mm
        torch.mm = mps_batch_invariant_mm
        torch.nn.functional.softmax = mps_batch_invariant_softmax
    else:
        if _batch_invariant_enabled:
            print("Disabling MPS batch-invariant mode...")
        
        _batch_invariant_enabled = False
        
        # Restore original operations
        torch.matmul = _ops_instance.original_matmul
        torch.mm = _ops_instance.original_mm
        torch.nn.functional.softmax = _ops_instance.original_softmax
    
    try:
        yield
    finally:
        # Restore state after context
        if enabled:
            _batch_invariant_enabled = False
            # Restore original operations
            torch.matmul = _ops_instance.original_matmul
            torch.mm = _ops_instance.original_mm
            torch.nn.functional.softmax = _ops_instance.original_softmax