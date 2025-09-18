#!/usr/bin/env python3
"""Download and cache model"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def download_model(model_name):
    print(f"Downloading {model_name}...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Download model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    print(f"✅ Model downloaded: {model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct"
    ]
    
    for model in models:
        download_model(model)