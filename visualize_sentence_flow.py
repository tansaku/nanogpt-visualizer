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
    """Create visualization showing how sentence flows through embedding dimensions using grid layout."""

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

    # Create visualization for each word separately
    word_images = []

    for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
        print(f"Creating grid for word '{word}' ({word_idx + 1}/{n_words})")

        # Create figure for this word with larger wordmaps
        wordmap_size = 4  # inches per wordmap
        fig_width = grid_cols * wordmap_size
        fig_height = grid_rows * wordmap_size

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))

        # Handle single row/column cases
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        elif grid_cols == 1:
            axes = axes.reshape(-1, 1)
        elif grid_rows == 1 and grid_cols == 1:
            axes = np.array([[axes]])

        # Fill the grid
        for dim in range(n_embd):
            row = dim // grid_cols
            col = dim % grid_cols
            ax = axes[row, col]

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
                opacity = max(0.15, min(0.95, opacity))
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
                    f"Dim {dim}\n{embed_value:.3f}",
                    fontsize=12,
                    ha="center",
                    va="center",
                    alpha=opacity,
                    color="black" if embed_value >= 0 else "red",
                    transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # Add dimension label and value
            ax.set_title(f"Dim {dim}\n{embed_value:.3f}", fontsize=12, pad=10)

            # Add border with color indicating positive/negative
            border_color = "black" if embed_value >= 0 else "red"
            border_width = 3 if abs(embed_value) > np.std(all_abs_values) else 1
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

        # Hide unused subplots
        for dim in range(n_embd, grid_rows * grid_cols):
            row = dim // grid_cols
            col = dim % grid_cols
            axes[row, col].set_visible(False)

        plt.suptitle(
            f'Word: "{word}" - Embedding Activations\n'
            f"Grid: {grid_rows}×{grid_cols} | Values: {embedding.min():.3f} to {embedding.max():.3f}",
            fontsize=18,
            y=0.98,
        )
        plt.tight_layout()

        # Save individual word visualization
        word_output_path = os.path.join(output_dir, f"word_{word_idx}_{word}.png")
        plt.savefig(word_output_path, dpi=200, bbox_inches="tight")
        plt.close()

        word_images.append(
            {
                "word": word,
                "filename": f"word_{word_idx}_{word}.png",
                "path": word_output_path,
            }
        )

    print(f"Saved {len(word_images)} word visualizations")
    return word_images


def generate_html_page(
    output_dir,
    model_name,
    probe_sentence,
    words,
    n_embd,
    word_embeddings=None,
    word_images=None,
):
    """Generate HTML page for the sentence flow visualization."""

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
            data_table_html += f"<th>Dim {dim}</th>"

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

    # Create word navigation and visualizations
    word_nav_html = ""
    word_visualizations_html = ""

    if word_images:
        # Navigation tabs
        word_nav_html = """
            <div class="word-navigation">
                <h3>Select Word to Visualize:</h3>
                <div class="word-tabs">"""

        for i, word_info in enumerate(word_images):
            active_class = "active" if i == 0 else ""
            word_nav_html += f"""
                <button class="word-tab {active_class}" onclick="showWord({i})">{word_info['word']}</button>"""

        word_nav_html += """
                </div>
            </div>"""

        # Word visualizations
        for i, word_info in enumerate(word_images):
            display_style = "block" if i == 0 else "none"
            word_visualizations_html += f"""
            <div class="word-visualization" id="word_{i}" style="display: {display_style};">
                <div class="zoom-container" id="zoomContainer_{i}">
                    <div class="zoom-controls">
                        <button class="zoom-btn" onclick="zoomIn({i})">+</button>
                        <button class="zoom-btn" onclick="zoomOut({i})">−</button>
                        <button class="zoom-btn" onclick="resetZoom({i})">Reset</button>
                        <button class="zoom-btn fullscreen-btn" onclick="toggleFullscreen({i})" id="fullscreenBtn_{i}">⛶ Fullscreen</button>
                    </div>
                    <img id="mainImage_{i}" src="{word_info['filename']}" alt="Word {word_info['word']} Visualization" style="width: 100%; height: auto;">
                </div>
            </div>"""

    # Calculate grid dimensions for display
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

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
            max-width: 1800px;
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
        .word-navigation {{
            margin: 20px 0;
            text-align: center;
        }}
        .word-tabs {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin: 15px 0;
        }}
        .word-tab {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            color: #495057;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .word-tab:hover {{
            background: #e9ecef;
            border-color: #adb5bd;
        }}
        .word-tab.active {{
            background: #667eea;
            border-color: #667eea;
            color: white;
        }}
        .word-visualization {{
            margin: 30px 0;
        }}
        .zoom-container {{
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: auto;
            position: relative;
            background: white;
            max-height: 85vh;
        }}
        .zoom-container.fullscreen {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 1000;
            max-height: none;
            border-radius: 0;
        }}
        .zoom-container img {{
            display: block;
            cursor: grab;
            transition: transform 0.1s;
            transform-origin: center center;
        }}
        .zoom-container img:active {{
            cursor: grabbing;
        }}
        .zoom-controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 1001;
            display: flex;
            gap: 8px;
        }}
        .zoom-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background 0.2s;
        }}
        .zoom-btn:hover {{
            background: #5a67d8;
        }}
        .fullscreen-btn {{
            background: #28a745;
        }}
        .fullscreen-btn:hover {{
            background: #218838;
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
        .info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
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
        .instructions {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentence Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <p><strong>Layout:</strong> {grid_rows}×{grid_cols} grid for {n_embd} dimensions</p>
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
                    <h3>Grid Layout</h3>
                    <div class="value">{grid_rows}×{grid_cols}</div>
                </div>
            </div>

            <div class="instructions">
                <strong>🔍 How to Use:</strong>
                <ul>
                    <li><strong>Word Tabs:</strong> Click on word tabs above to switch between different word visualizations</li>
                    <li><strong>Zoom:</strong> Use mouse wheel or +/- buttons to zoom in/out</li>
                    <li><strong>Pan:</strong> Click and drag to move around when zoomed in</li>
                    <li><strong>Fullscreen:</strong> Click fullscreen button for maximum viewing area</li>
                    <li><strong>Grid Layout:</strong> Each word shows a {grid_rows}×{grid_cols} grid of larger wordmaps</li>
                    <li><strong>Opacity:</strong> Wordmap opacity indicates activation strength for that dimension</li>
                    <li><strong>Borders:</strong> Thick borders indicate high activation, thin borders indicate low activation</li>
                </ul>
            </div>

            {word_nav_html}

            {word_visualizations_html}

            {data_table_html}

            <div class="info">
                <h3>Understanding the Grid Layout</h3>
                <p>Each word now has its own {grid_rows}×{grid_cols} grid showing all {n_embd} embedding dimensions:</p>
                <ul>
                    <li><strong>Each cell</strong> shows the wordmap for one dimension</li>
                    <li><strong>Opacity</strong> indicates how strongly that word activates that dimension</li>
                    <li><strong>Border thickness</strong> shows activation strength relative to other dimensions</li>
                    <li><strong>Colors:</strong> Black borders = positive activation, red borders = negative activation</li>
                    <li><strong>Grid layout</strong> makes it easier to see patterns and compare dimensions</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        let currentZoom = Array({len(words)}).fill(1);
        let isFullscreen = Array({len(words)}).fill(false);
        let currentWord = 0;

        function showWord(wordIndex) {{
            // Hide all word visualizations
            for (let i = 0; i < {len(words)}; i++) {{
                document.getElementById(`word_${{i}}`).style.display = 'none';
                document.querySelectorAll('.word-tab')[i].classList.remove('active');
            }}

            // Show selected word
            document.getElementById(`word_${{wordIndex}}`).style.display = 'block';
            document.querySelectorAll('.word-tab')[wordIndex].classList.add('active');
            currentWord = wordIndex;
        }}

        // Zoom functionality
        function setupZoomForWord(wordIndex) {{
            const container = document.getElementById(`zoomContainer_${{wordIndex}}`);
            const image = document.getElementById(`mainImage_${{wordIndex}}`);

            let isDragging = false;
            let startX, startY, startScrollLeft, startScrollTop;

            container.addEventListener('wheel', function(e) {{
                e.preventDefault();
                const rect = container.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                currentZoom[wordIndex] *= delta;
                currentZoom[wordIndex] = Math.max(0.3, Math.min(currentZoom[wordIndex], 15));

                // Calculate new scroll position to zoom toward mouse
                const scrollX = (container.scrollLeft + x) * delta - x;
                const scrollY = (container.scrollTop + y) * delta - y;

                image.style.transform = `scale(${{currentZoom[wordIndex]}})`;

                // Adjust scroll position after zoom
                setTimeout(() => {{
                    container.scrollLeft = scrollX;
                    container.scrollTop = scrollY;
                }}, 10);
            }});

            // Pan functionality
            container.addEventListener('mousedown', function(e) {{
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                startScrollLeft = container.scrollLeft;
                startScrollTop = container.scrollTop;
                container.style.cursor = 'grabbing';
            }});

            document.addEventListener('mousemove', function(e) {{
                if (!isDragging) return;
                e.preventDefault();
                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                container.scrollLeft = startScrollLeft - deltaX;
                container.scrollTop = startScrollTop - deltaY;
            }});

            document.addEventListener('mouseup', function() {{
                isDragging = false;
                container.style.cursor = 'grab';
            }});
        }}

        // Initialize zoom for all words
        for (let i = 0; i < {len(words)}; i++) {{
            setupZoomForWord(i);
        }}

        function zoomIn(wordIndex) {{
            currentZoom[wordIndex] *= 1.3;
            currentZoom[wordIndex] = Math.min(currentZoom[wordIndex], 15);
            document.getElementById(`mainImage_${{wordIndex}}`).style.transform = `scale(${{currentZoom[wordIndex]}})`;
        }}

        function zoomOut(wordIndex) {{
            currentZoom[wordIndex] *= 0.7;
            currentZoom[wordIndex] = Math.max(currentZoom[wordIndex], 0.3);
            document.getElementById(`mainImage_${{wordIndex}}`).style.transform = `scale(${{currentZoom[wordIndex]}})`;
        }}

        function resetZoom(wordIndex) {{
            currentZoom[wordIndex] = 1;
            const container = document.getElementById(`zoomContainer_${{wordIndex}}`);
            const image = document.getElementById(`mainImage_${{wordIndex}}`);
            image.style.transform = 'scale(1)';
            container.scrollLeft = 0;
            container.scrollTop = 0;
        }}

        function toggleFullscreen(wordIndex) {{
            const container = document.getElementById(`zoomContainer_${{wordIndex}}`);
            const btn = document.getElementById(`fullscreenBtn_${{wordIndex}}`);

            if (!isFullscreen[wordIndex]) {{
                container.classList.add('fullscreen');
                btn.textContent = '✕ Exit';
                isFullscreen[wordIndex] = true;
                document.body.style.overflow = 'hidden';
            }} else {{
                container.classList.remove('fullscreen');
                btn.textContent = '⛶ Fullscreen';
                isFullscreen[wordIndex] = false;
                document.body.style.overflow = 'auto';
            }}
        }}

        // Escape key to exit fullscreen
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape' && isFullscreen[currentWord]) {{
                toggleFullscreen(currentWord);
            }}
        }});
    </script>
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
    word_images = create_sentence_flow_visualization(
        embeddings, words, token_ids, itos, output_dir, probe_sentence, model_dir
    )

    # Generate HTML page
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        embeddings.shape[1],
        word_embeddings,
        word_images,
    )

    print(f"\nDone! Sentence flow visualization saved to: {output_dir}/")
    print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
