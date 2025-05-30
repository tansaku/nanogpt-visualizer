#!/usr/bin/env python3
"""
Positional Embedding Visualizer for NanoGPT

Generates wordmaps for each embedding dimension showing which sequence positions
are most strongly represented in that dimension.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv
import json

load_dotenv()


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return positional embeddings."""
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

    # Extract positional embeddings from model state
    model_state = checkpoint["model"]
    pos_embeddings = model_state["transformer.wpe.weight"]  # [block_size, n_embd]

    print(f"Positional embeddings shape: {pos_embeddings.shape}")

    return pos_embeddings, block_size, model_args


def create_position_wordcloud(dimension_values, dimension_idx, block_size):
    """Create a word cloud for a specific dimension using position numbers."""

    # Create position-value pairs
    position_weights = {}
    # Store actual values for data display
    dimension_data = {"positive": [], "negative": []}

    # Get absolute values for sizing (larger absolute values = bigger text)
    abs_values = np.abs(dimension_values)

    # Normalize to reasonable word cloud weights (1-100)
    if abs_values.max() > 0:
        normalized_weights = 1 + 99 * (abs_values / abs_values.max())
    else:
        normalized_weights = np.ones_like(abs_values)

    # Create position labels with weights and collect data
    for pos in range(block_size):
        position_weights[f"[{pos}]"] = normalized_weights[pos]

        # Collect data for HTML display
        value = dimension_values[pos]
        if value >= 0:
            dimension_data["positive"].append({"position": pos, "value": float(value)})
        else:
            dimension_data["negative"].append({"position": pos, "value": float(value)})

    # Sort by absolute value (highest magnitude first)
    dimension_data["positive"].sort(key=lambda x: abs(x["value"]), reverse=True)
    dimension_data["negative"].sort(key=lambda x: abs(x["value"]), reverse=True)

    # Only include positions with significant weights to avoid clutter
    # Keep top 50% of positions by absolute value
    threshold = np.percentile(abs_values, 50)
    significant_positions = {}
    for pos in range(block_size):
        if abs_values[pos] >= threshold:
            significant_positions[f"[{pos}]"] = normalized_weights[pos]

    if not significant_positions:
        # Fallback: include all positions
        significant_positions = position_weights

    # Create word cloud
    # Use a simple colormap based on positive/negative values
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        pos_num = int(word[1:-1])  # Extract number from [N] format
        value = dimension_values[pos_num]

        if value >= 0:
            # Positive values: shades of black (consistent with token visualizations)
            intensity = (
                min(255, int(255 * abs(value) / abs_values.max()))
                if abs_values.max() > 0
                else 128
            )
            gray_level = 255 - intensity  # 0 = black, 255 = white
            return f"rgb({gray_level}, {gray_level}, {gray_level})"
        else:
            # Negative values: shades of red
            intensity = (
                min(255, int(255 * abs(value) / abs_values.max()))
                if abs_values.max() > 0
                else 128
            )
            return f"hsl(0, 70%, {100 - intensity//4}%)"

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color="white",
        max_words=min(50, len(significant_positions)),
        color_func=color_func,
        prefer_horizontal=1.0,  # Force all text to be horizontal
        relative_scaling=0.5,
        min_font_size=8,
    ).generate_from_frequencies(significant_positions)

    return wordcloud, dimension_data


def visualize_positional_embeddings(pos_embeddings, block_size, model_args, output_dir):
    """Create and save position wordmaps for all embedding dimensions."""

    n_embd = pos_embeddings.shape[1]

    print(f"Creating position wordmaps for {n_embd} dimensions...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect dimension data for HTML
    all_dimension_data = {}

    # Generate wordmap for each dimension
    for dim in range(n_embd):
        print(f"Processing dimension {dim}/{n_embd}")

        # Get positional values for this dimension
        dimension_values = pos_embeddings[:, dim].numpy()

        # Create word cloud
        wordcloud, dimension_data = create_position_wordcloud(
            dimension_values, dim, block_size
        )
        all_dimension_data[dim] = dimension_data

        # Create plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # Add title with statistics
        value_range = f"{dimension_values.min():.3f} to {dimension_values.max():.3f}"
        plt.title(
            f"Positional Embedding Dimension {dim}\nValue Range: {value_range}",
            fontsize=14,
            pad=20,
        )

        # Save the plot
        output_path = os.path.join(output_dir, f"position_dimension_{dim}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        # Also save raw data for debugging
        if dim < 5:  # Save data for first few dimensions
            data_path = os.path.join(output_dir, f"position_dimension_{dim}_data.txt")
            with open(data_path, "w") as f:
                f.write(f"Dimension {dim} positional values:\n")
                for pos in range(block_size):
                    f.write(f"Position {pos}: {dimension_values[pos]:.6f}\n")

    print(f"Saved {n_embd} position wordmaps to: {output_dir}")
    return all_dimension_data


def generate_html_summary(
    output_dir, model_name, n_embd, block_size, all_dimension_data
):
    """Generate HTML page showing all positional wordmaps."""

    # Convert dimension data to JSON for embedding in HTML
    dimension_data_json = json.dumps(all_dimension_data)

    # Create grid of all dimensions
    grid_html = ""
    cols = 4  # Show 4 wordmaps per row

    for dim in range(n_embd):
        if dim % cols == 0:
            grid_html += "<div class='wordmap-row'>"

        grid_html += f"""
            <div class='wordmap-item' onclick="openModal({dim})">
                <h4>Dimension {dim}</h4>
                <img src='position_dimension_{dim}.png' alt='Position Dimension {dim}'>
            </div>"""

        if (dim + 1) % cols == 0 or dim == n_embd - 1:
            grid_html += "</div>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Positional Embeddings - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            line-height: 1.6;
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
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .content {{
            padding: 30px;
        }}
        .info {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .wordmap-row {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .wordmap-item {{
            flex: 1;
            min-width: 300px;
            text-align: center;
            border: 2px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .wordmap-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .wordmap-item h4 {{
            margin: 0;
            padding: 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }}
        .wordmap-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 20px;
            box-sizing: border-box;
        }}
        .modal-content {{
            background: white;
            border-radius: 8px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
        }}
        .modal .close {{
            position: absolute;
            top: 10px;
            right: 20px;
            color: #666;
            font-size: 30px;
            cursor: pointer;
            z-index: 1001;
        }}
        .modal-header {{
            padding: 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }}
        .modal-body {{
            padding: 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .modal-image {{
            flex: 1;
            min-width: 300px;
        }}
        .modal-image img {{
            width: 100%;
            border-radius: 4px;
        }}
        .modal-data {{
            flex: 1;
            min-width: 300px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .data-table th {{
            background: #28a745;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }}
        .data-table .positive {{
            color: #000;
        }}
        .data-table .negative {{
            color: #CC0000;
        }}
        .data-section {{
            margin-bottom: 20px;
        }}
        .data-section h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .button-group {{
            margin: 10px 0;
            text-align: center;
        }}
        .button-group button {{
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .button-group button:hover {{
            background: #218838;
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
            color: #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Positional Embeddings Visualization</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Block Size:</strong> {block_size} positions</p>
        </div>

        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h3>Dimensions</h3>
                    <div class="value">{n_embd}</div>
                </div>
                <div class="stat-card">
                    <h3>Max Positions</h3>
                    <div class="value">{block_size}</div>
                </div>
                <div class="stat-card">
                    <h3>Embeddings Type</h3>
                    <div class="value">Learned</div>
                </div>
            </div>

            <div class="info">
                <h3>Understanding Positional Embeddings</h3>
                <p>These visualizations show the <strong>learned positional embeddings</strong> from your NanoGPT model:</p>
                <ul>
                    <li><strong>Position Numbers:</strong> Each wordmap shows position labels ([0], [1], [2], etc.) indicating sequence positions</li>
                    <li><strong>Size indicates strength:</strong> Larger position numbers have stronger influence on that dimension</li>
                    <li><strong>Color indicates sign:</strong> Black for positive values, red for negative values</li>
                    <li><strong>Learned patterns:</strong> These aren't simple counting patterns - they're complex learned representations</li>
                </ul>
                <p><strong>Key insight:</strong> Unlike simple positional encoding, these embeddings learned which positions are most important for each semantic dimension during training.</p>
            </div>

            <h2>All Positional Embedding Dimensions</h2>
            <p>Click on any dimension to view the position wordmap and exact positional values</p>
            {grid_html}

            <div class="info">
                <h3>Next Steps</h3>
                <p>These positional wordmaps can be used with the sentence flow visualizer to show how sequence position affects representation.
                Each position in your input sequence will activate these learned positional patterns.</p>
            </div>
        </div>
    </div>

    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="close" onclick="closeModal()">&times;</span>
            <div class="modal-header">
                <h2 id="modal-title">Position Dimension X</h2>
                <div class="button-group">
                    <button onclick="showSection('image')">Position Wordmap</button>
                    <button onclick="showSection('data')">Exact Values</button>
                    <button onclick="showSection('both')">Both</button>
                </div>
            </div>
            <div class="modal-body">
                <div class="modal-image" id="modal-image-section">
                    <img id="modal-img" src="" alt="">
                </div>
                <div class="modal-data" id="modal-data-section">
                    <div class="data-section">
                        <h4>Highest Positive Values</h4>
                        <table class="data-table" id="positive-table">
                            <thead>
                                <tr><th>Position</th><th>Value</th></tr>
                            </thead>
                            <tbody id="positive-tbody"></tbody>
                        </table>
                    </div>
                    <div class="data-section">
                        <h4>Most Negative Values</h4>
                        <table class="data-table" id="negative-table">
                            <thead>
                                <tr><th>Position</th><th>Value</th></tr>
                            </thead>
                            <tbody id="negative-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dimensionData = {dimension_data_json};
        let currentView = 'both';

        function openModal(dimension) {{
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalTitle = document.getElementById('modal-title');

            modalImg.src = `position_dimension_${{dimension}}.png`;
            modalTitle.textContent = `Position Dimension ${{dimension}}`;

            loadDimensionData(dimension);
            showSection(currentView);
            modal.style.display = 'flex';
        }}

        function loadDimensionData(dimension) {{
            const data = dimensionData[dimension];
            if (!data) {{
                console.log('No data for dimension', dimension);
                return;
            }}

            const positiveTbody = document.getElementById('positive-tbody');
            const negativeTbody = document.getElementById('negative-tbody');

            positiveTbody.innerHTML = '';
            negativeTbody.innerHTML = '';

            if (data.positive) {{
                data.positive.forEach(item => {{
                    const row = positiveTbody.insertRow();
                    row.innerHTML = `<td class="positive">[${{item.position}}]</td><td class="positive">${{item.value.toFixed(6)}}</td>`;
                }});
            }}

            if (data.negative) {{
                data.negative.forEach(item => {{
                    const row = negativeTbody.insertRow();
                    row.innerHTML = `<td class="negative">[${{item.position}}]</td><td class="negative">${{item.value.toFixed(6)}}</td>`;
                }});
            }}
        }}

        function showSection(section) {{
            currentView = section;
            const imageSection = document.getElementById('modal-image-section');
            const dataSection = document.getElementById('modal-data-section');

            if (section === 'image') {{
                imageSection.style.display = 'block';
                dataSection.style.display = 'none';
            }} else if (section === 'data') {{
                imageSection.style.display = 'none';
                dataSection.style.display = 'block';
            }} else {{ // both
                imageSection.style.display = 'block';
                dataSection.style.display = 'block';
            }}
        }}

        function closeModal(event) {{
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('modal').style.display = 'none';
        }}

        // Close modal on escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>"""

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML summary: {html_path}")


def main():
    # Get configuration from environment
    checkpoint_path = os.environ.get("MODEL")

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    pos_embeddings, block_size, model_args = load_checkpoint(checkpoint_path)

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "positional_wordmaps")

    # Generate visualizations
    all_dimension_data = visualize_positional_embeddings(
        pos_embeddings, block_size, model_args, output_dir
    )

    # Generate HTML summary
    generate_html_summary(
        output_dir, model_dir, pos_embeddings.shape[1], block_size, all_dimension_data
    )

    print(f"\nDone! Positional embeddings visualization saved to: {output_dir}/")
    print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
