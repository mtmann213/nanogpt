#!/usr/bin/env python3
"""Environment verification script for nanoGPT follow-along."""

import torch
import sys

print("=" * 60)
print("nanoGPT Follow-Along — Environment Check")
print("=" * 60)

# PyTorch version
print(f"\nPyTorch: {torch.__version__}")
print(f"Python: {sys.version.split()[0]}")

# Device check  
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Quick GPU test
    try:
        x = torch.randn(64, 64, device='cuda')
        y = torch.matmul(x, x)
        print("GPU matmul test: PASS ✓")
        device = 'cuda'
    except Exception as e:
        print(f"GPU matmul test: FAIL ✗ ({e})")
        print("Falling back to CPU — training will work but be slower")
        device = 'cpu'
else:
    print("No GPU detected — using CPU")
    device = 'cpu'

# Dataset check
import os
data_dir = os.path.join(os.path.dirname(__file__), 'shakespeare_char')
input_file = os.path.join(data_dir, 'input.txt')
if os.path.exists(input_file):
    size_mb = os.path.getsize(input_file) / 1e6
    with open(input_file) as f:
        chars = set(f.read())
    print(f"\nDataset: Tiny Shakespeare ✓")
    print(f"  File: {input_file}")
    print(f"  Size: {size_mb:.2f} MB")  
    print(f"  Unique chars: {len(chars)} (vocab size)")
else:
    print(f"\nDataset NOT FOUND at {data_dir}")
    print("Run: python3 shakespeare_char/prepare.py")

# Dependencies check
for pkg in ['numpy', 'tqdm']:
    try:
        __import__(pkg)
        print(f"{pkg}: installed ✓")
    except ImportError:
        print(f"{pkg}: NOT installed — run: pip install {pkg}")

print("\n" + "=" * 60)
if device == 'cuda':
    print("Ready to train! Start with: python3 train.py")
else:
    print("Using CPU. For GPU, see README.md for ROCm troubleshooting.")
    print("Start with: python3 train.py (will use CPU)")
print("=" * 60)
