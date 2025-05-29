#!/usr/bin/env python3
"""
Sentence Flow Visualizer for NanoGPT

Visualizes how a probe sentence is represented across embedding dimensions.
Shows word-by-dimension matrix with opacity indicating embedding values.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from dotenv import load_dotenv
import re
from PIL import Image

load_dotenv()


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model configuration
    model_args = checkpoint["model_args"]
    vocab_size = model_args["vocab_size"]
    n_embd = model_args["n_embd"]

    print(f"Model vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {n_embd}")

    # Extract token embeddings from model state
    model_state = checkpoint["model"]
    embeddings = model_state["transformer.wte.weight"]  # [vocab_size, n_embd]

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings, vocab_size, model_args


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


def tokenize_sentence(sentence, stoi):
    """Tokenize a sentence using word tokenizer."""
    # Simple word tokenization (same as training)
    words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)

    token_ids = []
    for word in words:
        if word in stoi:
            token_ids.append(stoi[word])
        else:
            print(f"Warning: '{word}' not in vocabulary")

    return words, token_ids


def create_sentence_flow_visualization(
    embeddings, words, token_ids, itos, output_dir, probe_sentence, model_dir
):
    """Create visualization showing how sentence flows through embedding dimensions using actual wordmaps."""

    n_embd = embeddings.shape[1]
    n_words = len(words)

    print(
        f"Creating sentence flow visualization for {n_words} words across {n_embd} dimensions"
    )

    # Load existing wordmap images
    wordmap_dir = os.path.join("visualizations", model_dir, "embedding_wordmaps")
    if not os.path.exists(wordmap_dir):
        print(f"Error: Wordmap directory not found: {wordmap_dir}")
        print("Please run visualize_tokens.py first to generate embedding wordmaps")
        return None

    print(f"Loading wordmaps from: {wordmap_dir}")

    # Load wordmap images
    wordmap_images = {}
    for dim in range(n_embd):
        wordmap_path = os.path.join(wordmap_dir, f"dimension_{dim}.png")
        if os.path.exists(wordmap_path):
            wordmap_images[dim] = np.array(Image.open(wordmap_path))
        else:
            print(f"Warning: Missing wordmap for dimension {dim}")

    if not wordmap_images:
        print("Error: No wordmap images found!")
        return None

    # Get embedding values for each word
    word_embeddings = []
    for token_id in token_ids:
        if token_id < embeddings.shape[0]:
            word_embeddings.append(embeddings[token_id].numpy())
        else:
            word_embeddings.append(np.zeros(n_embd))

    # Normalize embeddings for opacity mapping
    all_values = np.array(word_embeddings).flatten()
    min_val, max_val = all_values.min(), all_values.max()
    print(f"Embedding value range: {min_val:.3f} to {max_val:.3f}")

    # Create a large figure
    cell_width = 4  # inches per cell
    cell_height = 3  # inches per cell
    fig_width = n_embd * cell_width
    fig_height = n_words * cell_height

    fig, axes = plt.subplots(n_words, n_embd, figsize=(fig_width, fig_height))

    # Handle case where we only have one word or one dimension
    if n_words == 1:
        axes = axes.reshape(1, -1)
    elif n_embd == 1:
        axes = axes.reshape(-1, 1)

    # Create visualization for each word-dimension pair
    for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
        for dim_idx in range(n_embd):
            ax = axes[word_idx, dim_idx]

            # Get the embedding value for this word in this dimension
            embed_value = embedding[dim_idx]

            # Normalize to [0, 1] for opacity (use absolute value for opacity)
            if max_val != min_val:
                opacity = abs(embed_value - min_val) / (max_val - min_val)
                # Ensure minimum visibility
                opacity = max(0.1, min(1.0, opacity))
            else:
                opacity = 0.5

            # Show the wordmap for this dimension
            if dim_idx in wordmap_images:
                wordmap_img = wordmap_images[dim_idx]

                # Apply color tint based on positive/negative value
                if embed_value >= 0:
                    # Positive: keep original colors but with opacity
                    ax.imshow(wordmap_img, alpha=opacity)
                else:
                    # Negative: apply red tint with opacity
                    red_tinted = wordmap_img.copy()
                    if len(red_tinted.shape) == 3:  # Color image
                        red_tinted[:, :, 0] = np.minimum(
                            255, red_tinted[:, :, 0] * 1.2
                        )  # Enhance red
                        red_tinted[:, :, 1] = red_tinted[:, :, 1] * 0.8  # Reduce green
                        red_tinted[:, :, 2] = red_tinted[:, :, 2] * 0.8  # Reduce blue
                    ax.imshow(red_tinted, alpha=opacity)
            else:
                # Fallback: show text
                ax.text(
                    0.5,
                    0.5,
                    f"Dim {dim_idx}\n{embed_value:.3f}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    alpha=opacity,
                    color="black" if embed_value >= 0 else "red",
                    transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Add value as title
            ax.set_title(f"{embed_value:.3f}", fontsize=10, pad=5)

            # Add border with color indicating positive/negative
            border_color = "black" if embed_value >= 0 else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

    # Add row and column labels
    for word_idx, word in enumerate(words):
        axes[word_idx, 0].set_ylabel(
            word, fontsize=14, rotation=0, ha="right", va="center", weight="bold"
        )

    for dim_idx in range(n_embd):
        axes[0, dim_idx].set_xlabel(f"Dimension {dim_idx}", fontsize=12, weight="bold")
        axes[0, dim_idx].xaxis.set_label_position("top")

    plt.suptitle(
        f'Sentence Flow Through Embedding Space\nProbe: "{probe_sentence}"\n'
        f"Wordmaps shown with opacity based on activation strength",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()

    # Save the visualization
    output_path = os.path.join(output_dir, "sentence_flow.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")  # Lower DPI for large images
    plt.close()

    print(f"Saved sentence flow visualization: {output_path}")
    return output_path


def generate_html_page(output_dir, model_name, probe_sentence, words, n_embd):
    """Generate HTML page for the sentence flow visualization."""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentence Flow Visualization - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .visualization {{
            text-align: center;
            margin: 30px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentence Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <p><strong>Embedding Dimensions:</strong> {n_embd}</p>
        </div>

        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h3>Words</h3>
                    <div class="value">{len(words)}</div>
                </div>
                <div class="stat-card">
                    <h3>Dimensions</h3>
                    <div class="value">{n_embd}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Cells</h3>
                    <div class="value">{len(words) * n_embd}</div>
                </div>
            </div>

            <div class="info">
                <h3>How to Read This Visualization</h3>
                <p>This visualization shows how each word in the probe sentence activates the learned embedding dimensions:</p>
                <ul>
                    <li><strong>Rows:</strong> Each word in the sentence ("{'" ", "'.join(words)}")</li>
                    <li><strong>Columns:</strong> Each embedding dimension (0 to {n_embd-1})</li>
                    <li><strong>Wordmaps:</strong> Each cell shows the learned word cloud for that dimension</li>
                    <li><strong>Opacity:</strong> How transparent/opaque the wordmap appears indicates activation strength</li>
                    <li><strong>Border Color:</strong> Black borders for positive activation, red for negative</li>
                    <li><strong>Numbers:</strong> The exact embedding value for that word-dimension pair</li>
                </ul>
                <p><em>Note: You must run visualize_tokens.py first to generate the embedding wordmaps.</em></p>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: black; border: 2px solid black;"></div>
                    <span>Positive Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: red; border: 2px solid red;"></div>
                    <span>Negative Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(0,0,0,0.3);"></div>
                    <span>Low Opacity = Weak Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(0,0,0,1);"></div>
                    <span>High Opacity = Strong Activation</span>
                </div>
            </div>

            <div class="visualization">
                <img src="sentence_flow.png" alt="Sentence Flow Visualization">
            </div>

            <div class="info">
                <h3>Interpretation</h3>
                <p>This visualization helps you understand:</p>
                <ul>
                    <li>Which dimensions are most activated by each word</li>
                    <li>How different words have different activation patterns</li>
                    <li>Whether certain dimensions specialize in certain types of words</li>
                    <li>The overall distribution of information across the embedding space</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>"""

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML page: {html_path}")


def main():
    # Get configuration from environment
    checkpoint_path = os.environ.get("MODEL")
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there bob")

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Using probe sentence: '{probe_sentence}'")

    # Load model and tokenizer
    embeddings, vocab_size, model_args = load_checkpoint(checkpoint_path)
    stoi, itos = load_tokenizer()

    if stoi is None:
        print("Failed to load tokenizer")
        sys.exit(1)

    # Tokenize probe sentence
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    print(f"Tokenized: {words} -> {token_ids}")

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "sentence_flow")
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualization
    create_sentence_flow_visualization(
        embeddings, words, token_ids, itos, output_dir, probe_sentence, model_dir
    )

    # Generate HTML page
    generate_html_page(
        output_dir, model_dir, probe_sentence, words, embeddings.shape[1]
    )

    print(f"\nDone! Sentence flow visualization saved to: {output_dir}/")
    print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
