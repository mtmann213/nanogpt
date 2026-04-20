"""
Stage 2-8 — GPT Transformer Architecture (Karpathy's video, ~25:00 - 1:05:00)

Built step-by-step following "Let's build GPT: from scratch, in code, spelled out"
Reference paper: "Attention Is All You Need" (Vaswani et al., 2017)

Architecture roadmap (follow along with video timestamps):
┌─────────────────────────────────────────────────────────┐
│  [~25:00] CausalSelfAttention                           │
│  [~40:00] MultiHeadAttention                            │  
│  [~48:00] Block (attention + feed-forward + norm)       │
│  [~55:00] GPT model (embeddings + blocks + head)        │
│  [~33:00] LayerNorm (pre-norm variant, added later)     │
└─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# STAGE 2 — Causal Self-Attention (~25:00 in video)
# "The neural network that models the sequence of words/characters"
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Single-head causal (masked) self-attention.
    
    For each position t, attends only to positions 0..t (past context).
    This is the key difference from bidirectional attention — it enforces
    autoregressive generation where you can't see future tokens.
    
    Architecture:
        X ──→ Linear → Q, K, V ──→ QK^T / sqrt(d_k) ──→ Mask ──→ Softmax ──→ Weights @ V ──→ Linear → Output
    
    Parameters:
        c_emb: embedding dimension (also the output dimension)  
        n_head: number of attention heads (1 for this single-head version)
    """
    
    def __init__(self, c_emb, n_head):
        super().__init__()
        assert c_emb % n_head == 0
        
        # Key, query, value transformations — each projects to c_emb dimensions
        # In the original paper these are separate W_Q, W_K, W_V matrices.
        # Karpathy combines them into one linear layer for efficiency:
        self.attn = nn.Linear(c_emb, 3 * c_emb, bias=False)
        
        # Output projection — maps concatenated head outputs back to c_emb
        self.proj = nn.Linear(c_emb, c_emb, bias=False)
        
        # Dropout for regularization (added later in video ~38:00)
        self.dropout = nn.Dropout(0.0)  # Set to 0.2 when scaling up
        
        # Cached causal mask — computed once, reused every forward pass
        # This triangular mask ensures position t can only attend to positions 0..t
        self.register_buffer('mask', torch.tril(torch.ones(n_head, n_head, n_head)).view(1, 1, n_head, n_head))
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, T, C) where:
               B = batch size, T = sequence length (block_size), C = embedding dim
        
        Returns:
            output tensor of same shape (B, T, C)
        """
        B, T, C = x.size()
        
        # Compute queries, keys, values all at once — more efficient than 3 separate calls
        q, k, v = self.attn(x).split(C, dim=-1)
        
        # Transpose to (B, n_head, T, head_size) for multi-head computation
        # We treat heads as a batch dimension here (single head case)
        k = k.view(B, T, C)      # (B, T, C)
        q = q.view(B, T, C)      # (B, T, C) 
        v = v.view(B, T, C)      # (B, T, C)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # For single head with C=embed_dim, this gives (B, T, T) matrix
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))  # (B, T, T)
        
        # Apply causal mask — zero out attention to future positions
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        att = F.softmax(att, dim=-1)
        
        # Dropout on attention weights (regularization — added later in video)
        att = self.dropout(att)
        
        # Weighted sum of values
        out = att @ v  # (B, T, C)
        
        # Output projection
        out = self.proj(out)
        
        return out


# ============================================================================
# STAGE 3 — Multi-Head Attention (~40:00 in video)
# "Multiple parallel attention heads, concatenated together"  
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention — batched implementation (like nanoGPT).
    
    Instead of looping over individual heads, we compute all heads in one 
    batched operation. This is mathematically equivalent but more efficient.
    
    Architecture:
        Input (B,T,C) → Linear(3*C) → split into Q,K,V → reshape to (B,nH,T,headSz)
                      → QK^T / sqrt(headSz) → mask → softmax → @V → concat heads → proj
    
    Each head operates on head_size = embed_dim // n_head dimensions.
    """
    
    def __init__(self, n_head, c_emb, block_size):
        super().__init__()
        self.n_head = n_head  # Save for use in forward()
        assert c_emb % n_head == 0
        head_size = c_emb // n_head
        
        # Single linear layer for Q,K,V projections — more efficient than 3 separate layers
        self.attn = nn.Linear(c_emb, 3 * c_emb, bias=False)
        
        # Output projection: concatenate all heads back to original dimension  
        self.proj = nn.Linear(c_emb, c_emb, bias=False)
        
        # Dropout for regularization (set to 0.2 when scaling up in video ~38:00)
        self.dropout = nn.Dropout(0.0)
        
        # Cached causal mask — computed once, reused every forward pass
        # Use a fixed max size; we slice it dynamically in forward() based on actual T
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, T, C) where:
               B = batch size, T = sequence length (block_size), C = embedding dim
        
        Returns:
            output tensor of same shape (B, T, C)
        """
        B, T, C = x.size()
        head_size = C // self.n_head
        
        # Compute Q, K, V all at once — (B, T, 3*C) then split
        q, k, v = self.attn(x).split(C, dim=-1)  # Each: (B, T, C)
        
        # Reshape to separate heads: (B, T, n_head, head_size) → transpose → (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nH, T, hS)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nH, T, hS)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)   # (B, nH, T, hS)
        
        # Compute attention scores: Q @ K^T / sqrt(head_size) → (B, nH, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (head_size ** 0.5))
        
        # Apply causal mask — zero out attention to future positions  
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # (B, nH, T, T)
        
        # Softmax to get attention weights
        att = F.softmax(att, dim=-1)
        
        # Dropout on attention weights (regularization — added later in video ~38:00)
        att = self.dropout(att)
        
        # Weighted sum of values → (B, nH, T, head_size)
        out = att @ v
        
        # Concatenate all heads back together → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.proj(out)
        
        return out


# ============================================================================  
# STAGE 4 — Feed-Forward Network (~48:00 in video)
# "Two linear layers with ReLU activation in between"
# ============================================================================

class FeedForward(nn.Module):
    """
    Simple two-layer feed-forward network applied per-token.
    
    Architecture: Linear → ReLU → Linear
    
    In the original Transformer paper, this inner dimension is 4x the 
    embedding dimension (512 → 2048). Karpathy adds this multiplier later.
    """
    
    def __init__(self, c_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_emb, 4 * c_emb),
            nn.ReLU(),
            nn.Linear(4 * c_emb, c_emb),
        )
        
    def forward(self, x):
        return self.net(x)


# ============================================================================
# STAGE 5 — Block: Attention + Feed-Forward + Residual + Norm (~48:00-52:00)
# "The fundamental building block of the Transformer"
# ============================================================================

class Block(nn.Module):
    """
    One layer of the Transformer stack.
    
    Architecture (pre-norm variant, from later in video ~35:00):
        X ──→ LayerNorm → MultiHeadAttention → Dropout → Add(X)
                  │
                LayerNorm → FeedForward → Dropout → Add(above)
    
    Pre-norm means LayerNorm is applied BEFORE the sub-layer (not after).
    This is a deviation from the original paper but has become standard.
    
    The residual connections (Add) are crucial — they allow gradients to flow
    through deep networks without vanishing.
    """
    
    def __init__(self, c_emb, n_head, block_size):
        super().__init__()
        # LayerNorm before each sub-layer (pre-norm formulation)
        self.ln1 = nn.LayerNorm(c_emb)
        self.ln2 = nn.LayerNorm(c_emb)
        
        # Multi-head attention sub-layer
        self.attn = MultiHeadAttention(n_head, c_emb, block_size)
        
        # Feed-forward sub-layer  
        self.ffwd = FeedForward(c_emb)
        
        # Dropout for regularization (set to 0.2 when scaling up)
        self.drop1 = nn.Dropout(0.0)
        self.drop2 = nn.Dropout(0.0)
        
    def forward(self, x):
        # Residual connection around multi-head attention
        x = x + self.drop1(self.attn(self.ln1(x)))
        
        # Residual connection around feed-forward  
        x = x + self.drop2(self.ffwd(self.ln2(x)))
        
        return x


# ============================================================================
# STAGE 6 — Full GPT Model (~55:00 in video)
# "Token embeddings + position embeddings → stacked blocks → final head"
# ============================================================================

class GPT(nn.Module):
    """
    Complete decoder-only Transformer (GPT architecture).
    
    Architecture flow:
        Input tokens ──→ Token Embedding ──+
                                            │
        Position indices ──→ Pos Embedding ─┼──→ [Block] × n_layer ──→ LayerNorm ──→ Linear ──→ Logits (vocab_size)
                                            │
    
    Key design choices:
    - Decoder-only (no encoder, no cross-attention) — suitable for language modeling/generation
    - Causal masking in attention ensures autoregressive property
    - Position embeddings provide token order information (unlike BERT which uses bidirectional attention)
    - Final linear layer projects from embedding space to vocabulary logits
    
    Parameters:
        vocab_size: number of unique tokens/characters  
        embed_dim: dimensionality of the embedding space (c_emb in papers)
        n_layer: number of Transformer blocks to stack
        n_head: number of attention heads per block
        block_size: maximum context length (sequence length)
    """
    
    def __init__(self, vocab_size, embed_dim=256, n_layer=6, n_head=8, block_size=256):
        super().__init__()
        
        self.block_size = block_size
        
        # Token embeddings — learnable lookup table: token_id → embedding vector
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings — learnable positional encoding  
        # Unlike sinusoidal positions in the original paper, GPT uses learned positions
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        
        # Stacked Transformer blocks
        self.blocks = nn.Sequential(*[Block(embed_dim, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        
        # Final layer normalization (before the output head)
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output head — projects from embedding space to vocabulary logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying: use same weights for input embeddings and output projection
        # (from GPT-2 paper — helps with training stability)
        self.token_embedding_table.weight = self.lm_head.weight
        
        # Initialize weights (standard Transformer initialization)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize all linear/embedding layers with standard normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.
        
        Args:
            idx: input token indices, shape (B, T) — batch of sequences
            targets: optional target tokens for computing loss, shape (B, T)
        
        Returns:
            If targets is None: logits of shape (B, T, vocab_size)
            If targets is provided: cross-entropy loss (scalar)
        """
        B, T = idx.shape
        
        # Validate sequence length
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        # Token embeddings
        token_emb = self.token_embedding_table(idx)  # (B, T, embed_dim)
        
        # Position embeddings — position i gets embedding at index i
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, embed_dim)
        
        # Add token + position embeddings
        x = token_emb + pos_emb  # (B, T, embed_dim)
        
        # Pass through Transformer blocks
        x = self.blocks(x)  # (B, T, embed_dim)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, embed_dim)
        
        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # If we have targets, compute loss
        if targets is not None:
            # Reshape for cross-entropy: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        
        # At inference time, just return logits
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive text generation.
        
        Given an initial sequence of tokens, repeatedly predict the next token
        and append it to the sequence until we've generated max_new_tokens.
        
        Args:
            idx: initial token indices, shape (B, T) — already encoded context
            max_new_tokens: how many new tokens to generate
            temperature: sampling temperature (lower = more deterministic, higher = more random)
            top_k: if set, only sample from the top-k most likely next tokens
        
        Returns:
            Complete sequence including original + generated tokens, shape (B, T+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size context (causal mask limitation)
            idx_cond = idx[:, -self.block_size:]
            
            # Get logits for the next token
            logits = self(idx_cond)  # (B, 1, vocab_size) — last position only
            
            # Take logits at the final time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Optional top-k filtering (only sample from top-k most likely tokens)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx


# ============================================================================
# HELPER — Count parameters
# ============================================================================

def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test with small config matching Karpathy's first training run
    vocab_size = 65  # Tiny Shakespeare vocabulary
    embed_dim = 32   # Small embedding for quick testing
    n_layer = 2      # Just 2 layers for speed
    n_head = 4
    block_size = 8   # Karpathy starts with block_size=8
    
    model = GPT(vocab_size, embed_dim, n_layer, n_head, block_size)
    
    print(f"Model architecture:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embedding dim: {embed_dim}")  
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  Block size: {block_size}")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Test forward pass with dummy data
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    logits = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {logits.shape}")
    
    # Test loss computation with targets
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    loss = model(x, targets=y)
    print(f"  Loss: {loss.item():.4f}")
    
    # Test generation
    generated = model.generate(x[:1], max_new_tokens=20, temperature=0.8)
    print(f"\nGeneration test:")
    print(f"  Generated shape: {generated.shape}")
