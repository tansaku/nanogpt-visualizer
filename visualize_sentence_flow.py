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
    """Create visualization showing how sentence flows through embedding dimensions using single vertical layout."""

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

    # Calculate grid dimensions (try to make it roughly square)
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))
    print(f"Using {grid_rows}×{grid_cols} grid for {n_embd} dimensions")

    # Create single figure with all words stacked vertically
    wordmap_size = 2.5  # inches per wordmap (smaller for tighter layout)
    fig_width = grid_cols * wordmap_size
    fig_height = n_words * grid_rows * wordmap_size

    print(f"Creating combined visualization: {fig_width:.1f}×{fig_height:.1f} inches")

    fig, axes = plt.subplots(
        n_words * grid_rows, grid_cols, figsize=(fig_width, fig_height)
    )

    # Handle single word case
    if n_words == 1:
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(grid_rows, -1)
    else:
        axes = axes.reshape(n_words * grid_rows, grid_cols)

    # Create visualization for each word
    for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
        print(f"Processing word '{word}' ({word_idx + 1}/{n_words})")

        # Fill the grid for this word
        for dim in range(n_embd):
            row_in_grid = dim // grid_cols
            col_in_grid = dim % grid_cols
            # Calculate absolute row position (word_idx * grid_rows + row_in_grid)
            abs_row = word_idx * grid_rows + row_in_grid
            ax = axes[abs_row, col_in_grid]

            # Get the embedding value for this word in this dimension
            embed_value = embedding[dim]

            # Normalize to [0, 1] for opacity (use percentile-based mapping for better contrast)
            if max_val != min_val:
                # Use percentile-based mapping to emphasize relative differences
                all_abs_values = np.abs(all_values)
                # Get percentile for this absolute value
                percentile = np.mean(all_abs_values <= abs(embed_value))
                # Apply power function to emphasize differences
                opacity = np.power(
                    percentile, 0.5
                )  # Square root to spread out lower values
                # Ensure minimum visibility and cap maximum
                opacity = max(0.2, min(0.95, opacity))
            else:
                opacity = 0.5

            # Show the wordmap for this dimension
            if dim in wordmap_images:
                wordmap_img = wordmap_images[dim]

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
                    f"D{dim}\n{embed_value:.2f}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    alpha=opacity,
                    color="black" if embed_value >= 0 else "red",
                    transform=ax.transAxes,
                )

            # Remove all ticks and spines for minimal appearance
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add dimension label only at the top row
            if word_idx == 0 and row_in_grid == 0:
                ax.set_title(f"D{dim}", fontsize=8, pad=2)

        # Hide unused subplots for this word
        for dim in range(n_embd, grid_rows * grid_cols):
            row_in_grid = dim // grid_cols
            col_in_grid = dim % grid_cols
            abs_row = word_idx * grid_rows + row_in_grid
            if abs_row < axes.shape[0]:
                axes[abs_row, col_in_grid].set_visible(False)

        # Add word label on the left side of the first row for this word
        if grid_cols > 0:
            first_row_for_word = word_idx * grid_rows
            axes[first_row_for_word, 0].text(
                -0.15,
                0.5,
                word,
                fontsize=14,
                fontweight="bold",
                rotation=0,
                ha="right",
                va="center",
                transform=axes[first_row_for_word, 0].transAxes,
            )

    # Minimize spacing
    plt.subplots_adjust(
        left=0.08,  # Space for word labels
        right=0.98,  # Minimal right margin
        top=0.95,  # Space for dimension labels
        bottom=0.02,  # Minimal bottom margin
        hspace=0.05,  # Minimal vertical spacing
        wspace=0.02,  # Minimal horizontal spacing
    )

    plt.suptitle(
        f'Sentence Flow: "{probe_sentence}"\n'
        f"Grid: {grid_rows}×{grid_cols} per word | Range: {min_val:.3f} to {max_val:.3f}",
        fontsize=14,
        y=0.98,
    )

    # Save the visualization
    output_path = os.path.join(output_dir, "sentence_flow.png")
    plt.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    print(f"Saved sentence flow visualization: {output_path}")
    return output_path


def generate_html_page(
    output_dir, model_name, probe_sentence, words, n_embd, word_embeddings=None
):
    """Generate simple HTML page for the sentence flow visualization."""

    # Calculate grid dimensions for display
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

    # Create data table if we have embeddings
    data_table_html = ""
    if word_embeddings is not None:
        data_table_html = f"""
            <div class="data-section">
                <h3>Exact Embedding Values</h3>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Word</th>"""

        for dim in range(n_embd):
            data_table_html += f"<th>D{dim}</th>"

        data_table_html += """
                            </tr>
                        </thead>
                        <tbody>"""

        for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
            data_table_html += f"<tr><td class='word-cell'><strong>{word}</strong></td>"
            for dim in range(n_embd):
                value = embedding[dim]
                cell_class = "positive" if value >= 0 else "negative"
                data_table_html += f"<td class='{cell_class}'>{value:.3f}</td>"
            data_table_html += "</tr>"

        data_table_html += """
                        </tbody>
                    </table>
                </div>
            </div>"""

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
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
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
            background: white;
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
        .info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .data-section {{
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }}
        .table-container {{
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        .data-table th {{
            background: #667eea;
            color: white;
            padding: 8px 4px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .data-table td {{
            padding: 6px 4px;
            text-align: center;
            border: 1px solid #eee;
        }}
        .word-cell {{
            background: #f8f9fa !important;
            font-weight: bold;
            position: sticky;
            left: 0;
            z-index: 5;
        }}
        .positive {{
            background: rgba(0, 0, 0, 0.05);
        }}
        .negative {{
            background: rgba(220, 0, 0, 0.05);
            color: #dc3545;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentence Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <p><strong>Layout:</strong> {grid_rows}×{grid_cols} grid per word, {len(words)} words vertically stacked</p>
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
                    <h3>Grid per Word</h3>
                    <div class="value">{grid_rows}×{grid_cols}</div>
                </div>
            </div>

            <div class="info">
                <h3>How to Read This Visualization</h3>
                <p>This visualization shows how each word in the probe sentence activates the learned embedding dimensions:</p>
                <ul>
                    <li><strong>Vertical Layout:</strong> Each word gets its own {grid_rows}×{grid_cols} grid, stacked vertically</li>
                    <li><strong>Word Labels:</strong> On the left side of each grid block</li>
                    <li><strong>Dimension Labels:</strong> At the top (D0, D1, D2, etc.)</li>
                    <li><strong>Wordmaps:</strong> Each cell shows the learned word cloud for that dimension</li>
                    <li><strong>Opacity:</strong> How strong the word activates that dimension</li>
                    <li><strong>Colors:</strong> Black tint for positive activation, red tint for negative activation</li>
                </ul>
                <p><strong>🔍 Usage:</strong> Use your browser's native zoom (Cmd/Ctrl + mouse wheel) to examine details</p>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(0,0,0,0.8);"></div>
                    <span>Strong Positive Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(220,0,0,0.8);"></div>
                    <span>Strong Negative Activation</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(0,0,0,0.2);"></div>
                    <span>Weak Activation</span>
                </div>
            </div>

            <div class="visualization">
                <img src="sentence_flow.png" alt="Sentence Flow Visualization">
            </div>

            {data_table_html}

            <div class="info">
                <h3>Interpretation Tips</h3>
                <p>Look for patterns in the visualization:</p>
                <ul>
                    <li><strong>Word Similarities:</strong> Do similar words (like "knock" "knock") have similar activation patterns?</li>
                    <li><strong>Dimension Specialization:</strong> Do certain dimensions consistently activate for specific types of words?</li>
                    <li><strong>Sentence Structure:</strong> How do different parts of the sentence (greeting vs. names) activate differently?</li>
                    <li><strong>Activation Strength:</strong> Which dimensions are most important for each word?</li>
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

    # Get embedding values for each word (for the data table)
    word_embeddings = []
    for token_id in token_ids:
        if token_id < embeddings.shape[0]:
            word_embeddings.append(embeddings[token_id].numpy())
        else:
            word_embeddings.append(np.zeros(embeddings.shape[1]))

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "sentence_flow")
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualization
    visualization_path = create_sentence_flow_visualization(
        embeddings, words, token_ids, itos, output_dir, probe_sentence, model_dir
    )

    if visualization_path:
        # Generate HTML page
        generate_html_page(
            output_dir,
            model_dir,
            probe_sentence,
            words,
            embeddings.shape[1],
            word_embeddings,
        )

        print(f"\nDone! Sentence flow visualization saved to: {output_dir}/")
        print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
