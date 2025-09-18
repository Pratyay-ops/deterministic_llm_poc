# quick_test.py

import sys
import os

# Test imports
try:
    from src import DeterministicMPSModel, check_mps_availability, get_m4_config
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test MPS availability
if check_mps_availability():
    print("✅ MPS is available")
else:
    print("⚠️  MPS not available, will use CPU")

# Test config
config = get_m4_config()
print(f"✅ Config loaded: {config['model']}")

# Test model initialization
try:
    print("Initializing model...")
    model = DeterministicMPSModel(config['model'])
    print("✅ Model initialized")
    
    # Test generation
    output = model.generate("Hello world", max_new_tokens=10)
    if isinstance(output, tuple):
        output = output[0]
    print(f"✅ Generation successful: {output[:50]}...")
    
except Exception as e:
    print(f"❌ Model initialization/generation failed: {e}")
    import traceback
    traceback.print_exc()