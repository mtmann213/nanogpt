"""
Stage 1 — Tokenizer (from Karpathy's video, ~9:00)

Character-level tokenizer for Tiny Shakespeare dataset.
Maps individual characters to integers and back.

Vocabulary size: 65 unique characters (space, newline, punctuation, a-z, A-Z)
"""

import torch
import os

# --- Configuration ---
data_dir = os.path.join(os.path.dirname(__file__), 'shakespeare_char')
input_file = os.path.join(data_dir, 'input.txt')

# --- Read the raw text ---
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Raw text length: {len(text)} characters")

# --- Build vocabulary ---
# Get all unique characters and sort them for consistent ordering
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# --- Create lookup tables ---
# stoi: string → integer (character to index)
# itos: integer → string (index to character)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# --- Encode function ---
def encode(s):
    """Encode a string into a list of integer token IDs."""
    return [stoi[ch] for ch in s]

# --- Decode function ---  
def decode(ids):
    """Decode a list of integer token IDs back into a string."""
    return ''.join([itos[i] for i in ids])

# --- Test the tokenizer ---
test_string = "Hello, world!"
encoded = encode(test_string)
decoded = decode(encoded)

print(f"\n--- Tokenizer Test ---")
print(f"Input:  '{test_string}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")
assert test_string == decoded, "Round-trip failed!"
print("Round-trip verification: PASS ✓")

# --- Encode the entire dataset into a PyTorch tensor ---
# This is what we'll feed into the Transformer for training
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nFull dataset as tensor:")
print(f"  Shape: {data.shape}")
print(f"  First 100 tokens: {data[:100].tolist()}")
print(f"  First 100 chars: '{decode(data[:100].tolist())}'")
