"""
Stage 10 — Text Generation / Sampling (Karpathy's video, ~1:27:00+)

Generate text from a trained GPT model with different sampling strategies.

Usage:
    python3 sample.py                          # Generate from final checkpoint  
    python3 sample.py --checkpoint path/to.pt   # Use specific checkpoint
    python3 sample.py --prompt "Hello"          # Start from custom prompt
"""

import torch
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model and its configuration."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    n_layer = config['n_layer']
    n_head = config['n_head']
    block_size = config['block_size']
    
    model = GPT(vocab_size, embed_dim, n_layer, n_head, block_size)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    return model


def load_vocabulary(data_dir):
    """Load the character vocabulary from the dataset."""
    input_file = os.path.join(data_dir, 'input.txt')
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def generate_text(model, prompt='', max_new_tokens=500, temperature=0.8, top_k=100):
    """Generate text starting from an optional prompt."""
    
    device = next(model.parameters()).device
    
    if prompt:
        # Encode the prompt
        chars = sorted(list(set(open(os.path.join(
            os.path.dirname(__file__), 'shakespeare_char', 'input.txt'
        )).read())))
        stoi = {ch: i for i, ch in enumerate(chars)}
        
        idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)
    else:
        # Start with empty context (model will generate from scratch)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate tokens autoregressively
    generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode to text
    itos = {i: ch for i, ch in enumerate(sorted(list(set(open(os.path.join(
        os.path.dirname(__file__), 'shakespeare_char', 'input.txt'
    )).read()))))}
    
    text = ''.join([itos[i.item()] for i in generated[0]])
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from a trained GPT model')
    parser.add_argument('--checkpoint', type=str, default='model_final.pt',
                        help='Path to model checkpoint (default: model_final.pt)')
    parser.add_argument('--prompt', type=str, default='',
                        help='Optional prompt to start generation from')
    parser.add_argument('--max-tokens', type=int, default=500,
                        help='Maximum number of tokens to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more deterministic, default: 0.8)')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Top-k filtering for sampling (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Loading model from: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Load model and vocabulary  
    data_dir = os.path.join(os.path.dirname(__file__), 'shakespeare_char')
    stoi, itos = load_vocabulary(data_dir)
    
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint '{args.checkpoint}' not found!")
        print("Train a model first with: python3 train.py")
        sys.exit(1)
    
    model = load_model(args.checkpoint, device)
    
    # Generate text
    print(f"\n{'='*60}")
    if args.prompt:
        print(f"Prompt: '{args.prompt}'")
    else:
        print("Starting from empty context (model generates from scratch)")
    print(f"Temperature: {args.temperature} | Top-k: {args.top_k}")
    print(f"{'='*60}\n")
    
    text = generate_text(
        model, 
        prompt=args.prompt, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Print output with line breaks for readability
    print(text[:2000])  # Limit output to first 2000 chars
    
    if len(text) > 2000:
        print(f"\n... ({len(text)} total characters generated)")
