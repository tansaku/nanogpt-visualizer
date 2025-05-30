#!/usr/bin/env python3
"""
Q/K/V Weight Matrix Visualizer for NanoGPT

Visualizes the linear transformation matrices that convert combined embeddings
into Query, Key, and Value spaces for attention computation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import json

load_dotenv()


def load_checkpoint_with_attention(checkpoint_path):
    """Load a NanoGPT checkpoint and extract attention matrices."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model configuration
    model_args = checkpoint["model_args"]
    vocab_size = model_args["vocab_size"]
    n_embd = model_args["n_embd"]
    n_head = model_args.get("n_head", 8)
    n_layer = model_args.get("n_layer", 6)

    print(f"Model vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {n_embd}")
    print(f"Number of heads: {n_head}")
    print(f"Number of layers: {n_layer}")

    # Extract model state
    model_state = checkpoint["model"]

    # Find attention weight matrices
    attention_layers = {}
    for key in model_state.keys():
        if "attn.c_attn" in key:
            # Extract layer information - handle patterns like "transformer.h.0.attn.c_attn.weight"
            parts = key.split(".")
            if "h" in parts:
                h_index = parts.index("h")
                if h_index + 1 < len(parts):
                    layer_num = parts[h_index + 1]
                    attention_layers[layer_num] = model_state[key]
                    print(
                        f"Found attention layer {layer_num}: {model_state[key].shape}"
                    )

    print(f"Found {len(attention_layers)} attention layers")

    return attention_layers, model_args


def split_qkv_weights(c_attn_weight, n_embd, n_head):
    """Split the combined QKV weight matrix into separate Q, K, V matrices."""
    print(f"c_attn_weight shape: {c_attn_weight.shape}")

    # Handle both orientations of the weight matrix
    if c_attn_weight.shape[0] == 3 * n_embd and c_attn_weight.shape[1] == n_embd:
        # Format: [3*n_embd, n_embd] - need to transpose for matrix multiplication
        print("Detected transposed QKV format [3*n_embd, n_embd]")
        c_attn_weight = c_attn_weight.T  # Now [n_embd, 3*n_embd]

    if c_attn_weight.shape[1] == 3 * n_embd:
        # Split into Q, K, V
        W_q = c_attn_weight[:, :n_embd]  # [n_embd, n_embd]
        W_k = c_attn_weight[:, n_embd : 2 * n_embd]  # [n_embd, n_embd]
        W_v = c_attn_weight[:, 2 * n_embd : 3 * n_embd]  # [n_embd, n_embd]

        print(f"Split QKV - Q: {W_q.shape}, K: {W_k.shape}, V: {W_v.shape}")
        return W_q, W_k, W_v
    else:
        print(f"Unexpected c_attn shape after processing: {c_attn_weight.shape}")
        print(f"Expected shape: [n_embd={n_embd}, 3*n_embd={3*n_embd}]")
        return None, None, None


def create_weight_heatmap(weight_matrix, matrix_type, layer_num, output_dir):
    """Create a heatmap visualization of a weight matrix."""
    print(f"Creating {matrix_type} heatmap for layer {layer_num}")

    # Convert to numpy if needed
    if isinstance(weight_matrix, torch.Tensor):
        weight_matrix = weight_matrix.numpy()

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        weight_matrix,
        cmap="RdBu_r",  # Red-Blue diverging colormap (red=positive, blue=negative)
        center=0,  # Center colormap at zero
        cbar_kws={"label": "Weight Value"},
        xticklabels=range(weight_matrix.shape[1]),
        yticklabels=range(weight_matrix.shape[0]),
    )

    plt.title(f"{matrix_type} Weight Matrix - Layer {layer_num}", fontsize=16, pad=20)
    plt.xlabel("Input Dimension", fontsize=14)
    plt.ylabel("Output Dimension", fontsize=14)

    # Add statistics to the plot
    mean_val = np.mean(weight_matrix)
    std_val = np.std(weight_matrix)
    min_val = np.min(weight_matrix)
    max_val = np.max(weight_matrix)

    stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(
        output_dir, f"{matrix_type.lower()}_layer_{layer_num}_heatmap.png"
    )
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return output_path, {
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
        "shape": weight_matrix.shape,
    }


def generate_qkv_html(output_dir, model_name, layer_data, model_args):
    """Generate HTML page for viewing Q/K/V weight matrix heatmaps."""

    n_embd = model_args["n_embd"]
    n_layers = len(layer_data)

    # Convert layer data to JSON for embedding in HTML
    layer_data_json = json.dumps(layer_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q/K/V Weight Matrices - {model_name}</title>
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
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
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
        .layer-tabs {{
            margin: 20px 0;
            text-align: center;
        }}
        .layer-tab-buttons {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin: 15px 0;
        }}
        .layer-tab-btn {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            color: #495057;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            min-width: 100px;
        }}
        .layer-tab-btn:hover {{
            background: #e9ecef;
            border-color: #adb5bd;
        }}
        .layer-tab-btn.active {{
            background: #9b59b6;
            border-color: #9b59b6;
            color: white;
        }}
        .qkv-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        .qkv-card {{
            border: 2px solid #ddd;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }}
        .qkv-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .qkv-card.query {{ border-color: #e74c3c; }}
        .qkv-card.key {{ border-color: #f39c12; }}
        .qkv-card.value {{ border-color: #27ae60; }}

        .qkv-header {{
            padding: 15px 20px;
            font-weight: bold;
            color: white;
            text-align: center;
        }}
        .qkv-header.query {{ background: #e74c3c; }}
        .qkv-header.key {{ background: #f39c12; }}
        .qkv-header.value {{ background: #27ae60; }}

        .qkv-image {{
            padding: 0;
        }}
        .qkv-image img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .qkv-stats {{
            padding: 15px 20px;
            background: #f8f9fa;
            font-size: 12px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
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
            max-width: 95%;
            max-height: 95%;
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
            text-align: center;
        }}
        .modal-body img {{
            max-width: 100%;
            height: auto;
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
            color: #9b59b6;
        }}
        .info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .layer-section {{
            display: none;
        }}
        .layer-section.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Q/K/V Weight Matrix Visualization</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Embedding Dimensions:</strong> {n_embd}×{n_embd} transformations</p>
            <p><strong>Layers:</strong> {n_layers} attention layers</p>
        </div>

        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h3>Matrix Size</h3>
                    <div class="value">{n_embd}×{n_embd}</div>
                </div>
                <div class="stat-card">
                    <h3>Attention Layers</h3>
                    <div class="value">{n_layers}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Parameters</h3>
                    <div class="value">{n_layers * 3 * n_embd * n_embd:,}</div>
                </div>
            </div>

            <div class="info">
                <h3>Understanding Q/K/V Weight Matrices</h3>
                <p>These heatmaps show the linear transformation matrices that convert combined embeddings into attention spaces:</p>
                <ul>
                    <li><strong>Query (Q) Matrix:</strong> How input dimensions combine to create "what I'm looking for" representations</li>
                    <li><strong>Key (K) Matrix:</strong> How input dimensions combine to create "what I offer" representations</li>
                    <li><strong>Value (V) Matrix:</strong> How input dimensions combine to create "what I contribute" representations</li>
                </ul>
                <p><strong>Color coding:</strong> Red = positive weights, Blue = negative weights, White = near zero</p>
                <p><strong>Click any matrix</strong> to view it in full size with detailed statistics</p>
            </div>

            <div class="layer-tabs">
                <h3>Attention Layer:</h3>
                <div class="layer-tab-buttons" id="layer-tabs"></div>
            </div>

            <div id="layer-sections"></div>
        </div>
    </div>

    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="close" onclick="closeModal()">&times;</span>
            <div class="modal-header">
                <h2 id="modal-title">Matrix Details</h2>
            </div>
            <div class="modal-body">
                <img id="modal-img" src="" alt="">
            </div>
        </div>
    </div>

    <script>
        const layerData = {layer_data_json};
        let currentLayer = Object.keys(layerData)[0];

        function generateLayerTabs() {{
            const tabsContainer = document.getElementById('layer-tabs');
            Object.keys(layerData).forEach((layer, index) => {{
                const button = document.createElement('button');
                button.className = `layer-tab-btn ${{index === 0 ? 'active' : ''}}`;
                button.textContent = `Layer ${{layer}}`;
                button.onclick = () => showLayer(layer);
                tabsContainer.appendChild(button);
            }});
        }}

        function generateLayerSections() {{
            const sectionsContainer = document.getElementById('layer-sections');

            Object.keys(layerData).forEach((layer, index) => {{
                const section = document.createElement('div');
                section.className = `layer-section ${{index === 0 ? 'active' : ''}}`;
                section.id = `layer-${{layer}}`;

                const qkvTypes = ['query', 'key', 'value'];
                const grid = document.createElement('div');
                grid.className = 'qkv-grid';

                qkvTypes.forEach(type => {{
                    const data = layerData[layer][type];
                    if (data) {{
                        const card = document.createElement('div');
                        card.className = `qkv-card ${{type}}`;
                        card.onclick = () => openModal(data.image, `${{type.toUpperCase()}} Matrix - Layer ${{layer}}`);

                        card.innerHTML = `
                            <div class="qkv-header ${{type}}">
                                ${{type.toUpperCase()}} Matrix
                            </div>
                            <div class="qkv-image">
                                <img src="${{data.image}}" alt="${{type}} matrix">
                            </div>
                            <div class="qkv-stats">
                                <div class="stat-row">
                                    <span>Mean:</span>
                                    <span>${{data.stats.mean.toFixed(4)}}</span>
                                </div>
                                <div class="stat-row">
                                    <span>Std:</span>
                                    <span>${{data.stats.std.toFixed(4)}}</span>
                                </div>
                                <div class="stat-row">
                                    <span>Range:</span>
                                    <span>[${{data.stats.min.toFixed(3)}}, ${{data.stats.max.toFixed(3)}}]</span>
                                </div>
                            </div>
                        `;

                        grid.appendChild(card);
                    }}
                }});

                section.appendChild(grid);
                sectionsContainer.appendChild(section);
            }});
        }}

        function showLayer(layer) {{
            // Update tabs
            document.querySelectorAll('.layer-tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');

            // Update sections
            document.querySelectorAll('.layer-section').forEach(section => {{
                section.classList.remove('active');
            }});
            document.getElementById(`layer-${{layer}}`).classList.add('active');

            currentLayer = layer;
        }}

        function openModal(imageSrc, title) {{
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalTitle = document.getElementById('modal-title');

            modalImg.src = imageSrc;
            modalTitle.textContent = title;
            modal.style.display = 'flex';
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

        // Initialize the page
        generateLayerTabs();
        generateLayerSections();
    </script>
</body>
</html>"""

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated Q/K/V HTML interface: {html_path}")


def main():
    # Get configuration from environment
    checkpoint_path = os.environ.get("MODEL")

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model and extract attention weights
    attention_layers, model_args = load_checkpoint_with_attention(checkpoint_path)

    if not attention_layers:
        print("No attention layers found in checkpoint!")
        sys.exit(1)

    # Create output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "qkv_weights")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating Q/K/V weight matrix visualizations...")
    print(f"Output directory: {output_dir}")

    n_embd = model_args["n_embd"]
    n_head = model_args.get("n_head", 8)

    # Process each attention layer
    layer_data = {}
    for layer_num, c_attn_weight in attention_layers.items():
        print(f"\nProcessing layer {layer_num}")

        # Split into Q, K, V matrices
        W_q, W_k, W_v = split_qkv_weights(c_attn_weight, n_embd, n_head)

        if W_q is None:
            print(f"Failed to extract Q/K/V matrices for layer {layer_num}")
            continue

        # Create heatmaps for each matrix type
        layer_data[layer_num] = {}

        matrices = [("Query", W_q), ("Key", W_k), ("Value", W_v)]
        for matrix_name, matrix in matrices:
            image_path, stats = create_weight_heatmap(
                matrix, matrix_name, layer_num, output_dir
            )
            layer_data[layer_num][matrix_name.lower()] = {
                "image": os.path.basename(image_path),
                "stats": stats,
            }

    # Generate HTML interface
    generate_qkv_html(output_dir, model_dir, layer_data, model_args)

    print(f"\nDone! Q/K/V weight visualizations saved to: {output_dir}/")
    print(f"Generated {len(attention_layers) * 3} heatmap images")
    print(f"View at: {output_dir}/index.html")


if __name__ == "__main__":
    main()
