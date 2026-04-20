"""
Stage 9-10 — Training Loop & Text Generation (Karpathy's video, ~1:05:00+)

Complete training pipeline for the GPT model on Tiny Shakespeare dataset.

Usage:
    python3 train.py                    # Train with default settings
    python3 train.py --device cuda      # Force GPU (if available)  
    python3 train.py --device cpu       # Force CPU
"""

import torch
import os
import sys
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn, optim
import argparse  # Added: CLI argument parsing for hyperparameter overrides

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, count_parameters


# ============================================================================
# CLI ARGUMENT PARSING — Override defaults from command line
# ============================================================================

parser = argparse.ArgumentParser(description='Train nanoGPT on Tiny Shakespeare')
parser.add_argument('--device', type=str, default='auto', help="'cuda', 'cpu', or 'auto'")
parser.add_argument('--block_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=384)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--max_steps', type=int, default=5000)
parser.add_argument('--eval_interval', type=int, default=250)
parser.add_argument('--eval_iters', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--log_interval', type=int, default=50, help='Print loss every N steps')
args = parser.parse_args()


# ============================================================================
# CONFIGURATION — Hyperparameters (start small, scale up later)
# ============================================================================

# Data settings
data_dir = os.path.join(os.path.dirname(__file__), 'shakespeare_char')
input_file = os.path.join(data_dir, 'input.txt')
block_size = args.block_size       # Context length — Karpathy starts with 8, scales to 256
batch_size = args.batch_size        # Mini-batch size — start small (4-8) for debugging

# Model architecture  
embed_dim = args.embed_dim        # Embedding dimension — start at 32 for testing
n_layer = args.n_layer             # Number of Transformer blocks
n_head = args.n_head               # Number of attention heads (embed_dim must be divisible by n_head)

# Training hyperparameters  
learning_rate = args.learning_rate # Learning rate
max_iters = args.max_steps         # Total training iterations
eval_interval = args.eval_interval # Evaluate every N iterations
eval_iters = args.eval_iters       # Number of batches for evaluation
device = args.device               # 'cuda', 'cpu', or 'auto' (try cuda, fall back to cpu)

# Dropout for regularization (added later in video when scaling up)
dropout = args.dropout             # Set to 0.0 for initial testing
log_interval = args.log_interval   # Print loss every N steps


# ============================================================================
# DATA LOADING — Tokenize and prepare training data
# ============================================================================

print("=" * 60)
print("nanoGPT Training — Loading Data")
print("=" * 60)

# Read raw text
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Raw text: {len(text):,} characters")

# Build vocabulary (character-level tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(f"Vocabulary: {vocab_size} unique characters")

# Encode entire dataset into integer tensor
encode = lambda s: [stoi[ch] for ch in s]
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded data: {len(data):,} tokens")

# Train/validation split (90/10) — Karpathy uses first 90% as train, last 10% as val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train set: {len(train_data):,} tokens")
print(f"Val set:   {len(val_data):,} tokens")


# ============================================================================
# DATA LOADER — Sample random chunks for training
# ============================================================================

def get_batch(split):
    """
    Get a random mini-batch of training or validation data.
    
    Each batch contains `batch_size` sequences, each of length `block_size + 1`.
    The input x is the first block_size tokens, and y (target) is the next token
    at each position — this creates multiple training examples packed into one chunk.
    
    For example with block_size=8:
        Input:  [t0, t1, t2, ..., t7]   → predict t1 given t0
                [t1, t2, t3, ..., t8]   → predict t2 given t1,t2  
                ...
    
    This is efficient because we train on contexts of length 1 through block_size
    simultaneously. It also helps the model learn to handle different context lengths
    during inference (where we might start with just 1 token).
    """
    data = train_data if split == 'train' else val_data
    
    # Sample random starting position in the dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract x (input) and y (target) for each sequence
    x = torch.stack([data[i:i+block_size] for i in ix])      # (B, T) — input context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # (B, T) — next token at each position
    
    return x, y


# ============================================================================
# DEVICE SETUP  
# ============================================================================

if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.randn(1, 1, embed_dim, device='cuda')
        y = torch.matmul(x.squeeze(), x.squeeze().T)
        print("GPU test: PASS ✓")
    except Exception as e:
        print(f"GPU test failed ({e}), falling back to CPU")
        device = 'cpu'

device = torch.device(device)


# ============================================================================
# MODEL INITIALIZATION  
# ============================================================================

print("\nInitializing model...")
model = GPT(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    n_layer=n_layer, 
    n_head=n_head,
    block_size=block_size
)
model.to(device)

total_params = count_parameters(model)
print(f"Model parameters: {total_params:,}")
print(f"Architecture:")
print(f"  Embedding dim: {embed_dim}")
print(f"  Layers: {n_layer}")  
print(f"  Heads: {n_head}")
print(f"  Head size: {embed_dim // n_head}")
print(f"  Block size: {block_size}")


# ============================================================================
# OPTIMIZER & TRAINING LOOP  
# ============================================================================

# Use AdamW optimizer (better than SGD for Transformers)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop — this is the core of what Karpathy builds in the video
print(f"\n{'='*60}")
print(f"Training for {max_iters} iterations")
print(f"{'='*60}\n")

for iter_num in range(max_iters):
    
    # --- Get a mini-batch ---
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    
    # --- Forward pass: compute loss ---
    logits = model(x)  # (B, T, vocab_size) — predictions for next token at each position
    
    # Cross-entropy loss: compare predicted logits vs actual next tokens
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # Flatten to (B*T, vocab_size)
        y.view(-1)                          # Flatten to (B*T,)
    )
    
    # --- Backward pass: compute gradients ---
    optimizer.zero_grad(set_to_none=True)  # Clear previous gradients
    loss.backward()                         # Compute gradients via autograd
    
    # --- Update weights ---
    optimizer.step()                        # Apply gradient descent update
    
    # --- Logging & evaluation ---
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for _ in range(eval_iters):
                xv, yv = get_batch('val')
                xv, yv = xv.to(device), yv.to(device)
                logits = model(xv)
                val_loss += F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    yv.view(-1)
                ).item()
        
        val_loss /= eval_iters
        
        # Print progress with validation loss
        print(f"Step {iter_num:5d} | Train loss: {loss.item():.4f} | Val loss: {val_loss:.4f}")
        
        # Generate a sample from the model
        model.eval()
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with empty context
        generated = model.generate(ctx, max_new_tokens=50, temperature=0.8, top_k=50)
        
        # Decode and print the generation  
        gen_text = ''.join([itos[i.item()] for i in generated[0]])
        print(f"  Sample: {gen_text[:100]}...")
        print()
        
        model.train()  # Back to training mode
    
    elif iter_num % log_interval == 0 or iter_num == max_iters - 1:
        # Print just loss (no eval/generation) at more frequent intervals
        print(f"Step {iter_num:5d} | Train loss: {loss.item():.4f}")
    
    # --- Save checkpoint every 500 steps ---
    if iter_num > 0 and iter_num % 500 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'iter_num': iter_num,
            'config': {
                'vocab_size': vocab_size, 'embed_dim': embed_dim,
                'n_layer': n_layer, 'n_head': n_head, 'block_size': block_size,
            }
        }, f'checkpoint_iter{iter_num}.pt')
        print(f"  Checkpoint saved: checkpoint_iter{iter_num}.pt\n")


# ============================================================================
# FINAL CHECKPOINT & GENERATION  
# ============================================================================

print("\nTraining complete! Saving final model...")
torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'iter_num': max_iters - 1,
    'config': {
        'vocab_size': vocab_size, 'embed_dim': embed_dim,
        'n_layer': n_layer, 'n_head': n_head, 'block_size': block_size,
    }
}, 'model_final.pt')

print("Final model saved to: model_final.pt")

# Generate a longer sample from the trained model
print("\n" + "=" * 60)
print("GENERATION — Let's see what our model produced!")
print("=" * 60)

model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

# Generate with different temperatures to explore the output space
for temp in [0.5, 0.8, 1.0, 1.2]:
    generated = model.generate(ctx, max_new_tokens=300, temperature=temp, top_k=100)
    gen_text = ''.join([itos[i.item()] for i in generated[0]])
    
    print(f"\n--- Temperature {temp} ---")
    # Print first 500 characters of generation  
    lines = gen_text[:500].split('\n')
    for line in lines:
        if line.strip():
            print(line)

print("\n" + "=" * 60)
print("Done! Compare your results with Karpathy's video.")
print("=" * 60)
