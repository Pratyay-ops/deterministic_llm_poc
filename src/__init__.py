# src/__init__.py
# Import the actual class names from your files
from .batch_invariant_ops_mps import set_mps_batch_invariant_mode, MPSBatchInvariantOps
from .deterministic_mps_inference import DeterministicMPSModel 
from .utils import get_m4_config, check_mps_availability
from .validation import validate_batch_invariance, validate_model_outputs
__version__ = "0.1.0"
__all__ = [
 "set_mps_batch_invariant_mode",
"MPSBatchInvariantOps",
"DeterministicMPSModel",
"get_m4_config",
"check_mps_availability",
"validate_batch_invariance",
"validate_model_outputs"
]