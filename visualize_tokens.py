#!/usr/bin/env python3
"""
Minimal NanoGPT Token Visualizer with Training Data Validation

Loads a NanoGPT checkpoint and creates word cloud visualizations of token embeddings.
Can validate which tokens are from training data vs. inherited from GPT-2.
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
    
    # For smaller vocabularies, use generic token names
    print("Using generic token IDs")
    return {i: f"token_{i}" for i in range(vocab_size)}

def analyze_training_data(training_data_path, vocab):
    """Analyze the training data to see which tokens actually appear."""
    print(f"\n📚 Analyzing training data: {training_data_path}")
    
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found: {training_data_path}")
        return None
    
    # Load and tokenize the training text
    with open(training_data_path, 'r', encoding='utf-8') as f:
        training_text = f.read()
    
    print(f"📝 Training text length: {len(training_text)} characters")
    print(f"📝 First 200 chars: {repr(training_text[:200])}")
    
    # Tokenize using GPT-2 tokenizer
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        # Allow the endoftext special token that appears in training data
        training_token_ids = enc.encode(training_text, allowed_special={"<|endoftext|>"})
        
        print(f"🔢 Training tokens: {len(training_token_ids)}")
        print(f"🔢 Unique training tokens: {len(set(training_token_ids))}")
        
        # Map back to our vocabulary indices
        training_vocab_indices = set()
        for token_id in set(training_token_ids):
            # For GPT-2 vocabulary, token_id should equal vocab index
            if token_id < len(vocab):
                training_vocab_indices.add(token_id)
        
        print(f"✅ Found {len(training_vocab_indices)} training tokens in model vocabulary")
        
        # Show sample training tokens
        sample_training_tokens = []
        for idx in list(training_vocab_indices)[:20]:
            sample_training_tokens.append(vocab.get(idx, f"token_{idx}"))
        print(f"📋 Sample training tokens: {sample_training_tokens}")
        
        return training_vocab_indices
        
    except Exception as e:
        print(f"❌ Failed to analyze training data: {e}")
        return None

def create_word_cloud(embeddings, vocab, dimension, output_dir, training_tokens=None, top_n=30):
    """Create a word cloud for a specific embedding dimension with positive (black) and negative (red) tokens."""
    print(f"Creating word cloud for dimension {dimension}")
    
    # Get values for this dimension
    dim_values = embeddings[:, dimension].numpy()
    
    # Get top positive tokens
    pos_indices = np.argsort(-dim_values)[:top_n]
    pos_scores = dim_values[pos_indices]
    
    # Get top negative tokens (by sorting the most negative values)
    neg_indices = np.argsort(dim_values)[:top_n]
    neg_scores = -dim_values[neg_indices]  # Make positive for word cloud frequencies
    
    # Create word frequencies and color mapping
    word_frequencies = {}
    word_colors = {}
    
    # Add positive tokens (black for training, gray for non-training)
    for idx, score in zip(pos_indices, pos_scores):
        if score > 0:  # Only truly positive values
            token = vocab.get(idx, f"token_{idx}")
            # Clean token string
            if token.strip() and len(token.strip()) > 0:
                word_frequencies[token] = float(score)
                if training_tokens is None:
                    word_colors[token] = '#000000'  # Black (no training data provided)
                elif idx in training_tokens:
                    word_colors[token] = '#000000'  # Black for training tokens
                else:
                    word_colors[token] = '#666666'  # Gray for non-training tokens
    
    # Add negative tokens (red for training, pink for non-training)
    for idx, score in zip(neg_indices, neg_scores):
        if dim_values[idx] < 0:  # Only truly negative values
            token = vocab.get(idx, f"token_{idx}")
            # Clean token string
            if token.strip() and len(token.strip()) > 0 and token not in word_frequencies:
                word_frequencies[token] = float(score)  # Use positive value for frequency
                if training_tokens is None:
                    word_colors[token] = '#CC0000'  # Red (no training data provided)
                elif idx in training_tokens:
                    word_colors[token] = '#CC0000'  # Red for training tokens
                else:
                    word_colors[token] = '#FF9999'  # Pink for non-training tokens
    
    if not word_frequencies:
        print(f"No valid tokens for dimension {dimension}")
        return
    
    print(f"Dimension {dimension}: {len([c for c in word_colors.values() if c == '#000000'])} positive (black), {len([c for c in word_colors.values() if c == '#CC0000'])} negative (red) tokens")
    
    if training_tokens is not None:
        # Count training vs non-training tokens
        training_count = len([c for c in word_colors.values() if c in ['#000000', '#CC0000']])
        non_training_count = len([c for c in word_colors.values() if c in ['#666666', '#FF9999']])
        print(f"  → {training_count} from training data, {non_training_count} NOT from training data")
        
        # Show examples of non-training tokens
        non_training_examples = [word for word, color in word_colors.items() if color in ['#666666', '#FF9999']][:10]
        if non_training_examples:
            print(f"  🔍 Non-training examples: {non_training_examples}")
    
    # Create word cloud with custom color function
    def color_func(word, **kwargs):
        return word_colors.get(word, '#000000')
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=len(word_frequencies),
        color_func=color_func,
        prefer_horizontal=1.0
    ).generate_from_frequencies(word_frequencies)
    
    # Save plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if training_tokens is not None:
        plt.title(f'Dimension {dimension} - Black/Red: Training tokens, Gray/Pink: Non-training', fontsize=14)
    else:
        plt.title(f'Dimension {dimension} - Black: Positive, Red: Negative', fontsize=14)
    
    output_path = os.path.join(output_dir, f'dimension_{dimension}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python visualize_tokens.py <checkpoint.pt> [training_data.txt]")
        print("Example: python visualize_tokens.py model.pt")
        print("Example: python visualize_tokens.py model.pt /path/to/training_data.txt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    training_data_path = sys.argv[2] if len(sys.argv) == 3 else None
    
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
        "/Users/samueljoseph/Documents/Github/tansaku/nanoGPT",
        os.path.expanduser("~/Documents/Github/karpathy/nanoGPT"),
        os.path.expanduser("~/Documents/Github/tansaku/nanoGPT"),
        os.path.expanduser("~/nanoGPT")
    ]
    for path in common_paths:
        if os.path.exists(path):
            nanogpt_path = path
            break
    
    vocab = create_vocabulary_mapping(vocab_size, nanogpt_path)
    
    # Analyze training data if provided
    training_tokens = None
    if training_data_path:
        training_tokens = analyze_training_data(training_data_path, vocab)
    
    # Create output directory
    output_dir = "token_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create word clouds for first few dimensions
    n_embd = embeddings.shape[1]
    dimensions_to_visualize = min(5, n_embd)  # First 5 dimensions or all if less
    
    print(f"Creating visualizations for {dimensions_to_visualize} dimensions...")
    
    for dim in range(dimensions_to_visualize):
        create_word_cloud(embeddings, vocab, dim, output_dir, training_tokens)
    
    print(f"\nDone! Visualizations saved to: {output_dir}/")
    print(f"Created {dimensions_to_visualize} word cloud images")
    
    if training_tokens is not None:
        print(f"\n🔍 ANALYSIS: Tokens in embeddings that are NOT from training data suggest:")
        print(f"   1. Model was fine-tuned from GPT-2 (retains original embeddings)")
        print(f"   2. OR model was trained from scratch but with full GPT-2 vocab")
        print(f"   3. Gray/Pink tokens show GPT-2's influence on your model")

if __name__ == "__main__":
    main()
