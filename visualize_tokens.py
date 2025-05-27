#!/usr/bin/env python3
"""
Minimal NanoGPT Token Visualizer

Loads a NanoGPT checkpoint and creates word cloud visualizations of token embeddings.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tiktoken

def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model configuration
    model_args = checkpoint['model_args']
    vocab_size = model_args['vocab_size']
    n_embd = model_args['n_embd']
    
    print(f"Model vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {n_embd}")
    
    # Extract token embeddings from model state
    model_state = checkpoint['model']
    embeddings = model_state['transformer.wte.weight']  # [vocab_size, n_embd]
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings, vocab_size, model_args

def create_vocabulary_mapping(vocab_size, nanogpt_path=None):
    """Create a mapping from model indices to token strings."""
    print(f"Creating vocabulary mapping for {vocab_size} tokens...")
    
    # If it's a standard GPT-2 sized vocabulary, use tiktoken directly
    if vocab_size >= 50000:
        print("Using standard GPT-2 vocabulary")
        enc = tiktoken.get_encoding("gpt2")
        gpt2_vocab_size = enc.n_vocab  # This should be 50257
        print(f"GPT-2 vocabulary size: {gpt2_vocab_size}")
        
        vocab = {}
        for i in range(vocab_size):
            if i < gpt2_vocab_size:
                try:
                    vocab[i] = enc.decode([i])
                except:
                    vocab[i] = f"<decode_error_{i}>"
            else:
                # Tokens beyond GPT-2 vocabulary - likely special tokens
                vocab[i] = f"<extra_token_{i}>"
        
        print(f"Mapped {len(vocab)} tokens ({gpt2_vocab_size} standard + {vocab_size - gpt2_vocab_size} extra)")
        return vocab
    
    # For smaller vocabularies, try to find training data
    if nanogpt_path:
        train_bin_path = os.path.join(nanogpt_path, "data/knock/train.bin")
        if os.path.exists(train_bin_path):
            return create_vocab_from_training_data(train_bin_path, vocab_size)
    
    # Fallback: generic token names
    print("Using generic token IDs")
    return {i: f"token_{i}" for i in range(vocab_size)}

def create_vocab_from_training_data(train_bin_path, expected_vocab_size):
    """Create vocabulary by analyzing training data."""
    print(f"Analyzing training data: {train_bin_path}")
    
    # Load training data
    train_data = np.frombuffer(open(train_bin_path, 'rb').read(), dtype=np.uint16)
    print(f"Training data: {len(train_data)} tokens")
    
    # Get unique tokens and their frequencies
    unique_tokens, counts = np.unique(train_data, return_counts=True)
    print(f"Unique tokens in training: {len(unique_tokens)}")
    print(f"Token range: {unique_tokens.min()} to {unique_tokens.max()}")
    
    # Sort by frequency (most frequent first)
    sorted_indices = np.argsort(-counts)
    tokens_by_frequency = unique_tokens[sorted_indices]
    
    print(f"Most frequent tokens: {tokens_by_frequency[:10].tolist()}")
    
    # Take only the number of tokens the model expects
    if len(tokens_by_frequency) > expected_vocab_size:
        tokens_by_frequency = tokens_by_frequency[:expected_vocab_size]
    
    # Decode tokens using tiktoken
    try:
        enc = tiktoken.get_encoding("gpt2")
        vocab = {}
        
        for model_idx, original_token_id in enumerate(tokens_by_frequency):
            try:
                token_str = enc.decode([int(original_token_id)])
                vocab[model_idx] = token_str
            except:
                vocab[model_idx] = f"token_{original_token_id}"
        
        print(f"Successfully decoded {len(vocab)} tokens")
        
        # Debug: show sample tokens
        sample = {k: v for k, v in list(vocab.items())[:20]}
        print(f"Sample tokens: {sample}")
        
        return vocab
        
    except Exception as e:
        print(f"Failed to decode tokens: {e}")
        return {i: f"token_{tokens_by_frequency[i]}" for i in range(len(tokens_by_frequency))}

def create_word_cloud(embeddings, vocab, dimension, output_dir, top_n=50):
    """Create a word cloud for a specific embedding dimension."""
    print(f"Creating word cloud for dimension {dimension}")
    
    # Get values for this dimension
    dim_values = embeddings[:, dimension].numpy()
    
    # Get top positive tokens
    top_indices = np.argsort(-dim_values)[:top_n]
    
    # Create word frequencies
    word_frequencies = {}
    for idx in top_indices:
        if dim_values[idx] > 0:  # Only positive values
            token = vocab.get(idx, f"token_{idx}")
            # Clean token string
            if token.strip() and len(token.strip()) > 0:
                word_frequencies[token] = dim_values[idx]
    
    if not word_frequencies:
        print(f"No valid tokens for dimension {dimension}")
        return
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=top_n,
        colormap='viridis'
    ).generate_from_frequencies(word_frequencies)
    
    # Save plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Dimension {dimension} - Top Tokens')
    
    output_path = os.path.join(output_dir, f'dimension_{dimension}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_tokens.py <checkpoint.pt>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load checkpoint
    embeddings, vocab_size, model_args = load_checkpoint(checkpoint_path)
    
    # Create vocabulary mapping
    # Try to find nanoGPT path automatically
    nanogpt_path = None
    common_paths = [
        "/Users/samueljoseph/Documents/Github/karpathy/nanoGPT",
        os.path.expanduser("~/Documents/Github/karpathy/nanoGPT"),
        os.path.expanduser("~/nanoGPT")
    ]
    for path in common_paths:
        if os.path.exists(path):
            nanogpt_path = path
            break
    
    vocab = create_vocabulary_mapping(vocab_size, nanogpt_path)
    
    # Create output directory
    output_dir = "token_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create word clouds for first few dimensions
    n_embd = embeddings.shape[1]
    dimensions_to_visualize = min(5, n_embd)  # First 5 dimensions or all if less
    
    print(f"Creating visualizations for {dimensions_to_visualize} dimensions...")
    
    for dim in range(dimensions_to_visualize):
        create_word_cloud(embeddings, vocab, dim, output_dir)
    
    print(f"\\nDone! Visualizations saved to: {output_dir}/")
    print(f"Created {dimensions_to_visualize} word cloud images")

if __name__ == "__main__":
    main()
