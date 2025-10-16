"""
CUDA Batch-Invariant Operations
Custom implementation using PyTorch operations with fixed-chunk processing
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
import os


def kahan_sum(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Kahan summation for numerical precision"""
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
    
    return sum_result.squeeze(dim) if not keepdim else sum_result


def batch_invariant_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Fixed-chunk matrix multiplication"""
    FIXED_CHUNK_SIZE = 32  # Warp-aligned for NVIDIA
    
    if len(input.shape) == 2:
        batch_size = input.shape[0]
        pad_size = (FIXED_CHUNK_SIZE - (batch_size % FIXED_CHUNK_SIZE)) % FIXED_CHUNK_SIZE
        
        if pad_size > 0:
            padded_input = F.pad(input, (0, 0, 0, pad_size), value=0.0)
        else:
            padded_input = input
        
        result = torch.matmul(padded_input, mat2)
        return result[:batch_size]
    else:
        return torch.matmul(input, mat2)


def batch_invariant_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with Kahan summation"""
    original_dtype = input.dtype
    input_double = input.double()
    
    max_vals = input_double.max(dim=dim, keepdim=True)[0]
    shifted = input_double - max_vals
    exp_vals = torch.exp(shifted)
    sum_exp = kahan_sum(exp_vals, dim=dim, keepdim=True)
    
    result = (exp_vals / sum_exp).to(original_dtype)
    return result


@contextmanager
def set_cuda_batch_invariant_mode(enabled: bool = True):
    """Context manager for CUDA batch-invariant operations"""
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    original_mm = torch.matmul
    original_softmax = F.softmax
    
    if enabled:
        # Set environment
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Configure PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Replace operations
        torch.matmul = batch_invariant_mm
        torch.mm = batch_invariant_mm
        F.softmax = batch_invariant_softmax
    
    try:
        yield
    finally:
        if enabled:
            torch.matmul = original_mm
            torch.mm = original_mm
            F.softmax = original_softmax
