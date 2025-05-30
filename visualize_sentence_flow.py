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
from PIL import Image, ImageDraw, ImageFont

load_dotenv()


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model configuration
    model_args = checkpoint["model_args"]
    vocab_size = model_args["vocab_size"]
    n_embd = model_args["n_embd"]
    block_size = model_args.get("block_size", 1024)

    print(f"Model vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {n_embd}")
    print(f"Block size: {block_size}")

    # Extract token embeddings and positional encodings from model state
    model_state = checkpoint["model"]
    token_embeddings = model_state["transformer.wte.weight"]  # [vocab_size, n_embd]
    pos_embeddings = model_state["transformer.wpe.weight"]  # [block_size, n_embd]

    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")

    return token_embeddings, pos_embeddings, vocab_size, model_args


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


class RepresentationHandler:
    """Handles different types of representations: embeddings, positional, combined."""

    def __init__(self, token_embeddings, pos_embeddings):
        self.token_embeddings = token_embeddings
        self.pos_embeddings = pos_embeddings
        self.n_embd = token_embeddings.shape[1]

    def get_representation(self, token_ids, representation_type="embeddings"):
        """Get representations for given token IDs."""
        word_representations = []

        for i, token_id in enumerate(token_ids):
            if token_id < self.token_embeddings.shape[0]:
                token_emb = self.token_embeddings[token_id].numpy()

                if representation_type == "embeddings":
                    word_representations.append(token_emb)
                elif representation_type == "positional":
                    if i < self.pos_embeddings.shape[0]:
                        pos_emb = self.pos_embeddings[i].numpy()
                        word_representations.append(pos_emb)
                    else:
                        word_representations.append(np.zeros(self.n_embd))
                elif representation_type == "combined":
                    if i < self.pos_embeddings.shape[0]:
                        pos_emb = self.pos_embeddings[i].numpy()
                        combined = token_emb + pos_emb
                        word_representations.append(combined)
                    else:
                        word_representations.append(token_emb)
                else:
                    raise ValueError(
                        f"Unknown representation type: {representation_type}"
                    )
            else:
                word_representations.append(np.zeros(self.n_embd))

        return word_representations


def create_sentence_flow_visualization(
    representation_handler,
    words,
    token_ids,
    itos,
    output_dir,
    probe_sentence,
    model_dir,
    representation_type="embeddings",
):
    """Create visualization by directly composing wordmap images with PIL - clean grids only."""

    n_embd = representation_handler.n_embd
    n_words = len(words)

    print(
        f"Creating {representation_type} visualization for {n_words} words across {n_embd} dimensions"
    )

    # Determine which wordmaps to load based on representation type
    if representation_type == "positional":
        wordmap_dir = os.path.join("visualizations", model_dir, "positional_wordmaps")
        wordmap_prefix = "position_dimension"
        print(f"Loading positional wordmaps from: {wordmap_dir}")
    else:
        wordmap_dir = os.path.join("visualizations", model_dir, "embedding_wordmaps")
        wordmap_prefix = "dimension"
        print(f"Loading token wordmaps from: {wordmap_dir}")

    if not os.path.exists(wordmap_dir):
        print(f"Error: Wordmap directory not found: {wordmap_dir}")
        if representation_type == "positional":
            print(
                "Please run visualize_positions.py first to generate positional wordmaps"
            )
        else:
            print("Please run visualize_tokens.py first to generate embedding wordmaps")
        return None

    # Load wordmap images
    wordmap_images = {}
    wordmap_size = None
    for dim in range(n_embd):
        wordmap_path = os.path.join(wordmap_dir, f"{wordmap_prefix}_{dim}.png")
        if os.path.exists(wordmap_path):
            img = Image.open(wordmap_path)
            wordmap_images[dim] = img
            if wordmap_size is None:
                wordmap_size = img.size  # (width, height)
        else:
            print(f"Warning: Missing wordmap for dimension {dim}")

    if not wordmap_images:
        print("Error: No wordmap images found!")
        return None

    print(f"Wordmap size: {wordmap_size}")

    # Get embedding values for each word using the representation handler
    word_embeddings = representation_handler.get_representation(
        token_ids, representation_type
    )

    # Normalize embeddings for opacity mapping
    all_values = np.array(word_embeddings).flatten()
    min_val, max_val = all_values.min(), all_values.max()
    print(f"Embedding value range: {min_val:.3f} to {max_val:.3f}")

    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))
    print(f"Using {grid_rows}×{grid_cols} grid for {n_embd} dimensions")

    # Calculate canvas size - much simpler without word labels
    wordmap_width, wordmap_height = wordmap_size

    # Make wordmaps smaller for overview
    small_wordmap_size = 60  # Slightly larger than before since no labels taking space
    scale_factor = small_wordmap_size / min(wordmap_width, wordmap_height)
    small_wordmap_width = int(wordmap_width * scale_factor)
    small_wordmap_height = int(wordmap_height * scale_factor)

    word_block_height = grid_rows * small_wordmap_height
    word_separator_height = 20  # Small gap between word blocks
    border_thickness = 2

    canvas_width = grid_cols * small_wordmap_width + border_thickness * 2
    canvas_height = (
        n_words * word_block_height
        + (n_words - 1) * word_separator_height
        + border_thickness * 2
    )

    print(f"Creating clean canvas: {canvas_width}×{canvas_height} pixels")
    print(
        f"Wordmaps resized from {wordmap_width}×{wordmap_height} to {small_wordmap_width}×{small_wordmap_height}"
    )

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Process each word
    for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
        print(f"Processing word '{word}' ({word_idx + 1}/{n_words})")

        # Calculate Y offset for this word's grid
        word_y_offset = border_thickness + word_idx * (
            word_block_height + word_separator_height
        )

        # Process each dimension for this word
        for dim in range(n_embd):
            if dim not in wordmap_images:
                continue

            # Calculate grid position
            row_in_grid = dim // grid_cols
            col_in_grid = dim % grid_cols

            # Calculate pixel position
            x = border_thickness + (col_in_grid * small_wordmap_width)
            y = word_y_offset + (row_in_grid * small_wordmap_height)

            # Get embedding value and calculate opacity
            embed_value = embedding[dim]

            # Use percentile-based mapping for better contrast (all representation types)
            if max_val != min_val:
                all_abs_values = np.abs(all_values)
                percentile = np.mean(all_abs_values <= abs(embed_value))
                opacity = np.power(percentile, 0.5)
                opacity = max(0.2, min(0.95, opacity))
            else:
                opacity = 0.5

            # Get the wordmap image and resize it
            wordmap_img = wordmap_images[dim].copy()
            wordmap_img = wordmap_img.resize(
                (small_wordmap_width, small_wordmap_height), Image.Resampling.LANCZOS
            )

            # Apply opacity and color effects based on positive/negative values
            if wordmap_img.mode != "RGBA":
                wordmap_img = wordmap_img.convert("RGBA")

            if embed_value < 0:
                # Negative values: invert colors to show suppression/inhibition
                img_array = np.array(wordmap_img)
                img_array[:, :, 0] = 255 - img_array[:, :, 0]  # Invert red
                img_array[:, :, 1] = 255 - img_array[:, :, 1]  # Invert green
                img_array[:, :, 2] = 255 - img_array[:, :, 2]  # Invert blue
                wordmap_img = Image.fromarray(img_array, "RGBA")

            # Apply opacity based on absolute value
            alpha = wordmap_img.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            wordmap_img.putalpha(alpha)

            # Paste onto canvas
            canvas.paste(wordmap_img, (x, y), wordmap_img)

    # Save the visualization
    output_path = os.path.join(output_dir, f"sentence_flow_{representation_type}.png")
    canvas.save(output_path, "PNG")

    print(f"Saved {representation_type} sentence flow visualization: {output_path}")
    return output_path, word_embeddings


def generate_html_page(
    output_dir, model_name, probe_sentence, words, n_embd, all_word_embeddings=None
):
    """Generate HTML page with tabs for different representation types."""

    # Calculate grid dimensions for display
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

    # Create word labels HTML for display alongside images
    words_html = ""
    for i, word in enumerate(words):
        words_html += f"""
            <div class="word-label">
                <span class="word-text">{word}</span>
                <span class="word-index">Word {i+1}</span>
            </div>"""

    # Create data tables for each representation type if we have embeddings
    data_tables_html = ""
    if all_word_embeddings:
        for rep_type, word_embeddings in all_word_embeddings.items():
            data_tables_html += f"""
                <div class="data-section" id="data_{rep_type}" style="display: none;">
                    <h3>Exact {rep_type.title()} Values</h3>
                    <div class="table-container">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Word</th>"""

            for dim in range(n_embd):
                data_tables_html += f"<th>D{dim}</th>"

            data_tables_html += """
                                </tr>
                            </thead>
                            <tbody>"""

            for word_idx, (word, embedding) in enumerate(zip(words, word_embeddings)):
                data_tables_html += (
                    f"<tr><td class='word-cell'><strong>{word}</strong></td>"
                )
                for dim in range(n_embd):
                    value = embedding[dim]
                    cell_class = "positive" if value >= 0 else "negative"
                    data_tables_html += f"<td class='{cell_class}'>{value:.3f}</td>"
                data_tables_html += "</tr>"

            data_tables_html += """
                            </tbody>
                        </table>
                    </div>
                </div>"""

    # Create visualization tabs
    visualization_tabs_html = (
        """
        <div class="representation-tabs">
            <h3>Representation Type:</h3>
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="showRepresentation('embeddings')">Token Embeddings</button>
                <button class="tab-btn" onclick="showRepresentation('positional')">Positional Encodings</button>
                <button class="tab-btn" onclick="showRepresentation('combined')">Combined (Token + Position)</button>
            </div>
        </div>

        <div class="visualizations-container">
            <div class="visualization" id="viz_embeddings" style="display: block;">
                <div class="viz-content">
                    <div class="word-labels">"""
        + words_html
        + """</div>
                    <div class="viz-image">
                        <img src="sentence_flow_embeddings.png" alt="Token Embeddings Visualization">
                    </div>
                </div>
            </div>
            <div class="visualization" id="viz_positional" style="display: none;">
                <div class="viz-content">
                    <div class="word-labels">"""
        + words_html
        + """</div>
                    <div class="viz-image">
                        <img src="sentence_flow_positional.png" alt="Positional Encodings Visualization">
                    </div>
                </div>
            </div>
            <div class="visualization" id="viz_combined" style="display: none;">
                <div class="viz-content">
                    <div class="word-labels">"""
        + words_html
        + """</div>
                    <div class="viz-image">
                        <img src="sentence_flow_combined.png" alt="Combined Representation Visualization">
                    </div>
                </div>
            </div>
        </div>"""
    )

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
            max-width: 1600px;
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
        .representation-tabs {{
            margin: 20px 0;
            text-align: center;
        }}
        .tab-buttons {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin: 15px 0;
        }}
        .tab-btn {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            color: #495057;
            padding: 15px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            min-width: 200px;
        }}
        .tab-btn:hover {{
            background: #e9ecef;
            border-color: #adb5bd;
        }}
        .tab-btn.active {{
            background: #667eea;
            border-color: #667eea;
            color: white;
        }}
        .visualizations-container {{
            margin: 30px 0;
        }}
        .visualization {{
            margin: 30px 0;
        }}
        .viz-content {{
            display: flex;
            gap: 30px;
            align-items: flex-start;
        }}
        .word-labels {{
            min-width: 200px;
            display: flex;
            flex-direction: column;
        }}
        .word-label {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 8px;
            border: 2px solid #dee2e6;
            text-align: center;
            /* Calculate height to match image rows */
            height: calc(({grid_rows} * 60px) - 4px); /* 60px per small wordmap, minus border adjustment */
            margin-bottom: 20px; /* Match word_separator_height from image generation */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .word-label:last-child {{
            margin-bottom: 0;
        }}
        .word-text {{
            display: block;
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .word-index {{
            display: block;
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .viz-image {{
            flex: 1;
            text-align: center;
        }}
        .viz-image img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
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
        .data-toggle {{
            text-align: center;
            margin: 20px 0;
        }}
        .data-toggle button {{
            background: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        .data-toggle button:hover {{
            background: #218838;
        }}

        /* Responsive design */
        @media (max-width: 1000px) {{
            .viz-content {{
                flex-direction: column;
            }}
            .word-labels {{
                min-width: unset;
                flex-direction: row;
                flex-wrap: wrap;
                justify-content: center;
            }}
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
                <h3>Understanding the Representations</h3>
                <p>This visualization shows three different representations of your probe sentence:</p>
                <ul>
                    <li><strong>Token Embeddings:</strong> The learned word representations from the vocabulary (using semantic wordmaps)</li>
                    <li><strong>Positional Encodings:</strong> Position-specific numerical patterns that encode sequence position (shown as numerical heatmap - these are the same for any words at these positions)</li>
                    <li><strong>Combined (Token + Position):</strong> What the model actually sees - token embeddings plus positional encodings (using wordmaps with combined values)</li>
                </ul>
                <p><strong>🔍 Usage:</strong> Switch between tabs to see how each representation affects the visualization. Use your browser's native zoom to examine details.</p>
                <p><strong>⚠️ Note:</strong> Positional encodings are shown as a numerical heatmap because they don't relate to word meanings - they're pure mathematical patterns based on sequence position.</p>
            </div>

            {visualization_tabs_html}

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

            <div class="data-toggle">
                <button onclick="toggleDataTables()" id="dataToggleBtn">Show Exact Values</button>
            </div>

            <div id="dataTables" style="display: none;">
                {data_tables_html}
            </div>

            <div class="info">
                <h3>Interpretation Tips</h3>
                <p>Compare the three representations to understand:</p>
                <ul>
                    <li><strong>Token vs Position:</strong> How much does word meaning vs word position contribute to each dimension?</li>
                    <li><strong>Positional Patterns:</strong> Do you see regular patterns in the positional encodings?</li>
                    <li><strong>Combined Effects:</strong> How does adding position change the token representations?</li>
                    <li><strong>Word Differences:</strong> How do identical words (like "knock" "knock") differ when position is added?</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        let currentRepresentation = 'embeddings';
        let dataVisible = false;

        function showRepresentation(repType) {{
            // Hide all visualizations
            document.querySelectorAll('.visualization').forEach(viz => {{
                viz.style.display = 'none';
            }});
            document.querySelectorAll('.data-section').forEach(data => {{
                data.style.display = 'none';
            }});

            // Remove active class from all tabs
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});

            // Show selected visualization
            document.getElementById(`viz_${{repType}}`).style.display = 'block';

            // Show corresponding data table if data is visible
            if (dataVisible) {{
                const dataSection = document.getElementById(`data_${{repType}}`);
                if (dataSection) {{
                    dataSection.style.display = 'block';
                }}
            }}

            // Add active class to clicked tab
            event.target.classList.add('active');
            currentRepresentation = repType;
        }}

        function toggleDataTables() {{
            const dataTables = document.getElementById('dataTables');
            const toggleBtn = document.getElementById('dataToggleBtn');

            if (dataVisible) {{
                dataTables.style.display = 'none';
                toggleBtn.textContent = 'Show Exact Values';
                dataVisible = false;
            }} else {{
                dataTables.style.display = 'block';
                toggleBtn.textContent = 'Hide Exact Values';
                // Show the current representation's data
                document.querySelectorAll('.data-section').forEach(data => {{
                    data.style.display = 'none';
                }});
                const currentData = document.getElementById(`data_${{currentRepresentation}}`);
                if (currentData) {{
                    currentData.style.display = 'block';
                }}
                dataVisible = true;
            }}
        }}
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
    embeddings, pos_embeddings, vocab_size, model_args = load_checkpoint(
        checkpoint_path
    )
    stoi, itos = load_tokenizer()

    if stoi is None:
        print("Failed to load tokenizer")
        sys.exit(1)

    # Tokenize probe sentence
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    print(f"Tokenized: {words} -> {token_ids}")

    # Create representation handler
    representation_handler = RepresentationHandler(embeddings, pos_embeddings)

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "sentence_flow")
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualization
    all_word_embeddings = {}
    for representation_type in ["embeddings", "positional", "combined"]:
        visualization_path, word_embeddings = create_sentence_flow_visualization(
            representation_handler,
            words,
            token_ids,
            itos,
            output_dir,
            probe_sentence,
            model_dir,
            representation_type,
        )
        all_word_embeddings[representation_type] = word_embeddings

    # Generate HTML page with all representations
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        embeddings.shape[1],
        all_word_embeddings,
    )

    print(f"\nDone! Sentence flow visualization saved to: {output_dir}/")
    print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
