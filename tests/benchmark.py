"""Basic usage example"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DeterministicMPSModel, get_m4_config

def main():
    # Get optimal config for your M4
    config = get_m4_config()
    print(f"Using model: {config['model']}")
    
    # Initialize model
    model = DeterministicMPSModel(config['model'])
    
    # Generate text
    prompts = [
        "Write a haiku about programming:",
        "Extract entities from: Apple Inc. CEO Tim Cook announced the M4 chip.",
        "def fibonacci(n):",
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        output, time_taken = model.generate(prompt, max_new_tokens=100)
        
        print(f"Output: {output}")
        print(f"Time: {time_taken:.3f}s")

if __name__ == "__main__":
    main()