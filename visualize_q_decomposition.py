#!/usr/bin/env python3
"""
Q Decomposition Visualizer for NanoGPT

Decomposes Q transformation dimensions into vocabulary word contributions.
Shows what each Q output dimension represents in terms of vocabulary words,
with opacity based on input word activation.
"""

import os
import sys
import torch
import numpy as np
import pickle
from dotenv import load_dotenv
import re
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

load_dotenv()


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    model_state = checkpoint["model"]

    token_embeddings = model_state["transformer.wte.weight"]  # [vocab_size, n_embd]
    pos_embeddings = model_state["transformer.wpe.weight"]  # [block_size, n_embd]

    print(f"Token embeddings: {token_embeddings.shape}")
    print(f"Positional embeddings: {pos_embeddings.shape}")

    return token_embeddings, pos_embeddings, model_args, model_state


def load_tokenizer():
    """Load the word tokenizer from meta file."""
    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if os.path.exists(meta_path):
        try:
            meta = pickle.load(open(meta_path, "rb"))
            if "itos" in meta and "stoi" in meta:
                print("Using word tokenizer from meta_word.pkl")
                return meta["stoi"], meta["itos"]
        except Exception as e:
            print(f"Failed to load meta_word.pkl: {e}")
    return None, None


def load_q_matrix(model_state, model_args, layer_idx=0):
    """Load Q weight matrix from the first attention layer."""
    print(f"Loading Q matrix from layer {layer_idx}")

    n_embd = model_args["n_embd"]
    attention_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"

    if attention_key not in model_state:
        print(f"Error: Attention layer {layer_idx} not found")
        return None

    c_attn_weight = model_state[attention_key]
    print(f"Found attention matrix: {c_attn_weight.shape}")

    # Handle transposed format if needed
    if c_attn_weight.shape[0] == 3 * n_embd and c_attn_weight.shape[1] == n_embd:
        print("Detected transposed QKV format [3*n_embd, n_embd]")
        c_attn_weight = c_attn_weight.T  # Now [n_embd, 3*n_embd]

    if c_attn_weight.shape[1] == 3 * n_embd:
        # Extract just the Q matrix
        W_q = c_attn_weight[:, :n_embd].numpy()  # [n_embd, n_embd]
        print(f"Extracted Q matrix: {W_q.shape}")
        return W_q
    else:
        print(f"Unexpected c_attn shape: {c_attn_weight.shape}")
        return None


def get_word_representation(word, stoi, token_embeddings, pos_embeddings, position=0):
    """Get the complete representation of a word (token + positional)."""
    if word not in stoi:
        print(f"Warning: '{word}' not in vocabulary")
        return None

    token_id = stoi[word]

    # Get token embedding
    token_emb = token_embeddings[token_id].numpy()  # [n_embd]

    # Get positional embedding
    if position < pos_embeddings.shape[0]:
        pos_emb = pos_embeddings[position].numpy()  # [n_embd]
    else:
        pos_emb = np.zeros_like(token_emb)

    # Combined representation
    combined_repr = token_emb + pos_emb  # [n_embd]

    print(f"Word '{word}' representation:")
    print(f"  Token embedding range: [{token_emb.min():.3f}, {token_emb.max():.3f}]")
    print(f"  Positional embedding range: [{pos_emb.min():.3f}, {pos_emb.max():.3f}]")
    print(f"  Combined range: [{combined_repr.min():.3f}, {combined_repr.max():.3f}]")

    return combined_repr


def compute_q_activations(word_repr, W_q):
    """Compute Q activations for the word representation."""
    q_activations = word_repr @ W_q  # [n_embd] @ [n_embd, n_embd] = [n_embd]

    print(
        f"Q activations range: [{q_activations.min():.3f}, {q_activations.max():.3f}]"
    )
    print(
        f"Q activations mean: {q_activations.mean():.3f}, std: {q_activations.std():.3f}"
    )

    return q_activations


def compute_vocab_contributions_for_q_dim(token_embeddings, W_q, q_dim, itos):
    """Compute how much each vocabulary word contributes to a specific Q output dimension."""
    print(f"Computing vocabulary contributions for Q dimension {q_dim}")

    vocab_size = token_embeddings.shape[0]
    n_embd = token_embeddings.shape[1]

    # Get the Q weights for this output dimension
    q_weights_for_dim = W_q[
        :, q_dim
    ]  # [n_embd] - weights from each input dim to this Q output dim

    vocab_contributions = {}

    for token_id in range(vocab_size):
        if token_id < len(itos):
            word = itos[token_id]

            # Get token embedding for this word
            token_emb = token_embeddings[token_id].numpy()  # [n_embd]

            # Calculate contribution: sum over input dims of (embedding[input_dim] * q_weight[input_dim])
            contribution = np.sum(token_emb * q_weights_for_dim)

            vocab_contributions[word] = float(contribution)

    print(f"Computed contributions for {len(vocab_contributions)} vocabulary words")

    # Show top contributors
    sorted_contribs = sorted(
        vocab_contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )
    print(f"Top 10 contributors (by absolute value):")
    for i, (word, contrib) in enumerate(sorted_contribs[:10]):
        print(f"  {i+1:2d}. {word:15s}: {contrib:8.4f}")

    return vocab_contributions


def create_q_dimension_wordcloud(
    vocab_contributions, q_activation, q_dim, output_dir, word
):
    """Create a wordcloud for a Q dimension showing vocabulary contributions."""

    # Filter out very small contributions to avoid clutter, but be less restrictive
    min_contribution_threshold = 0.0001  # Much lower threshold
    filtered_contributions = {
        word: contrib
        for word, contrib in vocab_contributions.items()
        if abs(contrib) >= min_contribution_threshold
    }

    if not filtered_contributions:
        print(f"No significant contributions for Q dimension {q_dim}")
        return None

    print(
        f"Creating wordcloud for Q dimension {q_dim} with {len(filtered_contributions)} words"
        f" (out of {len(vocab_contributions)} total vocabulary words)"
    )

    # Separate positive and negative contributions
    positive_contributions = {w: c for w, c in filtered_contributions.items() if c > 0}
    negative_contributions = {w: c for w, c in filtered_contributions.items() if c < 0}

    print(f"  Positive contributors: {len(positive_contributions)}")
    print(f"  Negative contributors: {len(negative_contributions)}")

    # Normalize contributions for wordcloud sizing
    max_abs_contrib = max(abs(c) for c in filtered_contributions.values())

    # Create two separate wordclouds for positive and negative contributions
    wordcloud_width = 800
    wordcloud_height = 300  # Smaller height since we'll stack them

    # Create positive wordcloud (black text)
    if positive_contributions:
        pos_frequencies = {}
        for word, contrib in positive_contributions.items():
            pos_frequencies[word] = abs(contrib) / max_abs_contrib * 100

        pos_wordcloud = WordCloud(
            width=wordcloud_width,
            height=wordcloud_height,
            background_color="white",
            max_words=200,  # More words allowed
            color_func=lambda *args, **kwargs: "black",  # All black for positive
            relative_scaling=0.5,
            min_font_size=6,
        ).generate_from_frequencies(pos_frequencies)

        pos_img = pos_wordcloud.to_image()
    else:
        pos_img = Image.new("RGB", (wordcloud_width, wordcloud_height), "white")

    # Create negative wordcloud (red text)
    if negative_contributions:
        neg_frequencies = {}
        for word, contrib in negative_contributions.items():
            neg_frequencies[word] = abs(contrib) / max_abs_contrib * 100

        neg_wordcloud = WordCloud(
            width=wordcloud_width,
            height=wordcloud_height,
            background_color="white",
            max_words=200,  # More words allowed
            color_func=lambda *args, **kwargs: "red",  # All red for negative
            relative_scaling=0.5,
            min_font_size=6,
        ).generate_from_frequencies(neg_frequencies)

        neg_img = neg_wordcloud.to_image()
    else:
        neg_img = Image.new("RGB", (wordcloud_width, wordcloud_height), "white")

    # Combine the two wordclouds vertically
    combined_height = wordcloud_height * 2 + 40  # Space for separator
    combined_img = Image.new("RGB", (wordcloud_width, combined_height), "white")

    # Paste positive wordcloud on top
    combined_img.paste(pos_img, (0, 0))

    # Add separator line and label
    draw = ImageDraw.Draw(combined_img)
    separator_y = wordcloud_height + 10
    draw.line(
        [(50, separator_y), (wordcloud_width - 50, separator_y)], fill="gray", width=2
    )

    try:
        separator_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        separator_font = ImageFont.load_default()

    draw.text(
        (50, separator_y + 5),
        "Positive contributions (black) ↑",
        fill="black",
        font=separator_font,
    )
    draw.text(
        (50, separator_y + 20),
        "Negative contributions (red) ↓",
        fill="red",
        font=separator_font,
    )

    # Paste negative wordcloud on bottom
    combined_img.paste(neg_img, (0, wordcloud_height + 40))

    # Apply opacity based on Q activation strength
    max_possible_activation = max(abs(q_activation), 0.1)  # Avoid division by zero
    opacity = min(1.0, abs(q_activation) / max_possible_activation)
    opacity = max(0.2, opacity)  # Ensure minimum visibility

    # Apply opacity
    if combined_img.mode != "RGBA":
        combined_img = combined_img.convert("RGBA")

    # Apply opacity by modifying alpha channel
    alpha = (
        combined_img.split()[-1]
        if combined_img.mode == "RGBA"
        else Image.new("L", combined_img.size, 255)
    )
    alpha = alpha.point(lambda p: int(p * opacity))
    combined_img.putalpha(alpha)

    # Create final image with label and stats
    label_height = 100
    final_width = wordcloud_width
    final_height = combined_height + label_height

    final_img = Image.new("RGB", (final_width, final_height), "white")

    # Paste combined wordcloud
    final_img.paste(combined_img, (0, 0), combined_img)

    # Add labels
    draw = ImageDraw.Draw(final_img)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title = f"Q Dimension {q_dim} - Word: {word}"
    draw.text((10, combined_height + 10), title, fill="black", font=font_large)

    # Stats
    stats_text = f"Activation: {q_activation:.4f} | Opacity: {opacity:.2f} | Max contrib: {max_abs_contrib:.4f}"
    draw.text((10, combined_height + 40), stats_text, fill="gray", font=font_small)

    # Contribution counts
    contrib_text = f"Pos: {len(positive_contributions)} words | Neg: {len(negative_contributions)} words | Total: {len(filtered_contributions)}/{len(vocab_contributions)}"
    draw.text((10, combined_height + 60), contrib_text, fill="gray", font=font_small)

    # Color indicator for overall activation direction
    color = "green" if q_activation >= 0 else "red"
    activation_text = (
        f"Overall: {'Positive' if q_activation >= 0 else 'Negative'} Activation"
    )
    draw.text((10, combined_height + 80), activation_text, fill=color, font=font_small)

    # Save
    output_path = os.path.join(output_dir, f"q_dim_{q_dim:02d}_{word}.png")
    final_img.save(output_path, "PNG")

    print(f"Saved Q dimension {q_dim} wordcloud: {output_path}")
    return output_path


def create_overview_grid(output_paths, word, n_embd, output_dir):
    """Create a grid overview of all Q dimension wordclouds."""

    if not output_paths:
        return None

    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

    # Load first image to get size
    first_img = Image.open(output_paths[0])
    img_width, img_height = first_img.size

    # Scale down for overview
    scale_factor = 0.3
    thumb_width = int(img_width * scale_factor)
    thumb_height = int(img_height * scale_factor)

    # Calculate canvas size
    spacing = 10
    canvas_width = grid_cols * thumb_width + (grid_cols - 1) * spacing + 2 * spacing
    canvas_height = (
        grid_rows * thumb_height + (grid_rows - 1) * spacing + 2 * spacing + 60
    )

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Add title
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()

    title = f"Q Decomposition Overview - Word: {word}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 10), title, fill="black", font=title_font)

    # Place thumbnails
    for i, path in enumerate(output_paths):
        if i >= n_embd:
            break

        row = i // grid_cols
        col = i % grid_cols

        x = spacing + col * (thumb_width + spacing)
        y = 60 + row * (thumb_height + spacing)

        # Load and resize image
        img = Image.open(path)
        thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        canvas.paste(thumb, (x, y))

        # Add dimension label
        dim_label = f"D{i}"
        label_x = x + thumb_width // 2 - 10
        label_y = y + thumb_height - 20
        draw.text((label_x, label_y), dim_label, fill="white", font=title_font)

    # Save overview
    overview_path = os.path.join(output_dir, f"q_decomposition_overview_{word}.png")
    canvas.save(overview_path, "PNG")

    print(f"Saved overview: {overview_path}")
    return overview_path


def main():
    # Get configuration
    checkpoint_path = os.environ.get("MODEL")
    target_word = os.environ.get("TARGET_WORD", "knock")
    position = int(os.environ.get("WORD_POSITION", "0"))

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    print(f"Analyzing word: '{target_word}' at position {position}")

    # Load model and tokenizer
    token_embeddings, pos_embeddings, model_args, model_state = load_checkpoint(
        checkpoint_path
    )
    stoi, itos = load_tokenizer()

    if stoi is None:
        print("Failed to load tokenizer")
        sys.exit(1)

    # Load Q matrix
    W_q = load_q_matrix(model_state, model_args, layer_idx=0)
    if W_q is None:
        print("Failed to load Q matrix")
        sys.exit(1)

    # Get word representation (token + positional)
    word_repr = get_word_representation(
        target_word, stoi, token_embeddings, pos_embeddings, position
    )
    if word_repr is None:
        sys.exit(1)

    # Compute Q activations
    q_activations = compute_q_activations(word_repr, W_q)

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join(
        "visualizations", model_dir, f"q_decomposition_{target_word}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating Q decomposition visualizations...")
    print(f"Output directory: {output_dir}")

    # Generate wordcloud for each Q dimension
    n_embd = model_args["n_embd"]
    output_paths = []

    for q_dim in range(n_embd):
        print(f"\nProcessing Q dimension {q_dim}/{n_embd-1}")

        # Compute vocabulary contributions for this Q dimension
        vocab_contributions = compute_vocab_contributions_for_q_dim(
            token_embeddings, W_q, q_dim, itos
        )

        # Create wordcloud with opacity based on activation
        output_path = create_q_dimension_wordcloud(
            vocab_contributions, q_activations[q_dim], q_dim, output_dir, target_word
        )

        if output_path:
            output_paths.append(output_path)

    # Create overview grid
    overview_path = create_overview_grid(output_paths, target_word, n_embd, output_dir)

    print(f"\nDone! Q decomposition visualization saved to: {output_dir}/")
    print(f"Key outputs:")
    print(f"  - Individual Q dimensions: q_dim_*.png")
    print(f"  - Overview grid: {overview_path}")

    print(f"\n📊 Summary:")
    print(f"  - Analyzed word '{target_word}' at position {position}")
    print(f"  - Generated {len(output_paths)} Q dimension wordclouds")
    print(
        f"  - Q activation range: [{q_activations.min():.3f}, {q_activations.max():.3f}]"
    )
    print(
        f"  - Strongest Q activation: dimension {np.argmax(np.abs(q_activations))} = {q_activations[np.argmax(np.abs(q_activations))]:.3f}"
    )


if __name__ == "__main__":
    main()
