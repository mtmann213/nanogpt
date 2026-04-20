# Let's Build GPT — Follow-Along Workspace

Based on Andrej Karpathy's video: "Let's build GPT: from scratch, in code, spelled out" (Oct 2023)

## Quick Start

```bash
cd ~/hermes/nano-gpt/follow-along

# Verify environment
python3 check_env.py

# Follow along with the video — see README.md for step-by-step guide
cat README.md
```

## Project Structure

```
follow-along/
├── shakespeare_char/       # Tiny Shakespeare dataset (pre-downloaded)
│   ├── input.txt           # Raw text (~1MB, all of Shakespeare)
│   ├── train.bin           # Pre-tokenized training data
│   └── val.bin             # Pre-tokenized validation data  
├── tokenizer.py            # Character-level tokenizer (Stage 1)
├── model.py                # Transformer architecture (Stages 2-8, built incrementally)
├── train.py                # Training loop (Stages 9-10)
├── sample.py               # Text generation / sampling (final stage)
├── requirements.txt        # Python dependencies
├── check_env.py            # Environment verification script
└── README.md               # This file — step-by-step guide
```

## GPU Status

**Note:** Your Corsair has an AMD Radeon 8060S iGPU (gfx1151). PyTorch ROCm build may have kernel compatibility issues with this specific chip. The code will work on CPU as a fallback — training will be slower but fully functional for all stages of the video.

Check your device:
```bash
python3 check_env.py
```

If GPU doesn't work, set `device='cpu'` in train.py (or use `'auto'` which tries cuda then falls back to cpu).

## Step-by-Step Guide (Matches Video Timeline)

### Stage 1 — Tokenizer (~9:00 in video)
File: `tokenizer.py`
- Character-level vocabulary (65 chars)
- encode() and decode() functions
- Convert text → integer sequences and back

### Stage 2 — Data Loading (~13:00)  
Inline in train.py or separate data module
- Load input.txt, tokenize everything into a single tensor
- Train/val split (90%/10%)
- Block sampling with context length `block_size`

### Stage 3 — Causal Self-Attention (~25:00)
File: `model.py` — add `CausalSelfAttention` class
- Q, K, V projections  
- Dot-product attention with triangular mask
- Output projection

### Stage 4 — Multi-Head Attention (~40:00)
File: `model.py` — extend to `MultiHeadAttention`
- Multiple parallel heads concatenated
- Still single-head version for clarity first

### Stage 5 — Feed-Forward Network (~48:00)
File: `model.py` — add `Block` class  
- Multi-head attention + feed-forward + residual connections + layer norm

### Stage 6 — Full Transformer (~55:00)
File: `model.py` — add `GPT` class
- Token embeddings + position embeddings
- Stacked blocks
- Final layer norm + linear head → vocabulary logits

### Stage 7 — Training Loop (~1:05:00)  
File: `train.py`
- Forward pass, loss computation (cross-entropy)
- Backward pass via PyTorch autograd
- SGD optimizer with learning rate
- Training loop over epochs/batches

### Stage 8 — Scaling Up (~1:20:00)
Increase hyperparameters: embed_dim=384, n_layer=6, n_head=6, block_size=256
Add dropout for regularization
Add LayerNorm (pre-norm variant from later in video ~33:00)

### Stage 9 — Text Generation (~1:27:00)
File: `sample.py` or inline function
- Autoregressive sampling from the trained model
- Temperature control for randomness

## Reference Implementation

The final, complete nanoGPT repo is at:
```bash
cd ~/hermes/nano-gpt/nanoGPT
# Contains production-ready train.py + model.py (~300 lines each)
# Can reproduce GPT-2 124M on OpenWebText
```

Use `nanoGPT/` as a reference to compare against your incremental builds.

## Expected Results at Each Stage

| Model Size | Params | Vocab | Tokens | Val Loss | GPU Time (A100) | CPU Time (Corsair) |
|-----------|--------|-------|--------|----------|-----------------|-------------------|
| Tiny (block_size=8) | ~50K | 65 | 300K | ~4.0+ | seconds | minutes |  
| Small (embed=256, layer=6) | ~1M | 65 | 300K | ~2.5 | ~5 min | ~30-60 min |
| Medium (embed=384, layer=6) | ~10M | 65 | 300K | ~1.48 | ~15 min | ~2-4 hours |

## Tips

- Start small! The block_size=8 model trains in seconds on CPU — great for verifying your code works before scaling up
- Use `device='cpu'` if GPU has issues, or `'auto'` to try both
- Each commit in the video corresponds to adding a new class/function to model.py
- The nanoGPT repo uses batched multi-head attention (more efficient than Karpathy's step-by-step version)
