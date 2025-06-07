#!/usr/bin/env python3
"""
Q/K/V Decomposition Visualizer for NanoGPT

Decomposes Q, K, and V transformation dimensions into vocabulary word contributions.
Shows what each dimension represents in terms of vocabulary words.
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


def load_qkv_matrices(model_state, model_args, layer_idx=0):
    """Load Q, K, V weight matrices from the first attention layer."""
    print(f"Loading Q/K/V matrices from layer {layer_idx}")

    n_embd = model_args["n_embd"]
    attention_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"

    if attention_key not in model_state:
        print(f"Error: Attention layer {layer_idx} not found")
        return None, None, None

    c_attn_weight = model_state[attention_key]
    print(f"Found attention matrix: {c_attn_weight.shape}")

    # Handle transposed format if needed
    if c_attn_weight.shape[0] == 3 * n_embd and c_attn_weight.shape[1] == n_embd:
        print("Detected transposed QKV format [3*n_embd, n_embd]")
        c_attn_weight = c_attn_weight.T  # Now [n_embd, 3*n_embd]

    if c_attn_weight.shape[1] == 3 * n_embd:
        # Extract Q, K, V matrices
        W_q = c_attn_weight[:, :n_embd].numpy()
        W_k = c_attn_weight[:, n_embd : 2 * n_embd].numpy()
        W_v = c_attn_weight[:, 2 * n_embd :].numpy()
        print(f"Extracted Q: {W_q.shape}, K: {W_k.shape}, V: {W_v.shape}")
        return W_q, W_k, W_v
    else:
        print(f"Unexpected c_attn shape: {c_attn_weight.shape}")
        return None, None, None


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


def compute_transformed_activations(word_repr, W_matrix):
    """Compute Q, K, or V activations for the word representation."""
    activations = word_repr @ W_matrix  # [n_embd] @ [n_embd, n_embd] = [n_embd]
    return activations


def compute_vocab_contributions_for_dim(token_embeddings, W_matrix, dim, itos):
    """Compute how much each vocabulary word contributes to a specific output dimension."""
    q_weights_for_dim = W_matrix[:, dim]

    vocab_contributions = {}

    for token_id in range(token_embeddings.shape[0]):
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


def create_dimension_wordcloud(
    vocab_contributions, activation, dim, output_dir, word, matrix_name, stoi
):
    """Create a wordcloud for a dimension showing vocabulary contributions."""

    print(f"\n=== DEBUG {matrix_name} Dimension {dim} ===")
    print(f"{matrix_name} activation: {activation:.6f}")
    print(f"Total vocab contributions computed: {len(vocab_contributions)}")

    # Show contribution statistics
    all_contribs = list(vocab_contributions.values())
    if all_contribs:
        print(f"Contribution range: [{min(all_contribs):.6f}, {max(all_contribs):.6f}]")
        print(f"Contribution mean: {np.mean(all_contribs):.6f}")
        print(
            f"Non-zero contributions: {sum(1 for c in all_contribs if abs(c) > 1e-10)}"
        )

    # Start with a very low threshold, we can always increase it later
    min_contribution_threshold = 1e-6  # Very permissive threshold
    filtered_contributions = {
        word: contrib
        for word, contrib in vocab_contributions.items()
        if abs(contrib) >= min_contribution_threshold
    }

    print(
        f"After filtering (threshold {min_contribution_threshold}): {len(filtered_contributions)} words"
    )

    # If still too few words, take the top contributors regardless of threshold
    if len(filtered_contributions) < 20:
        print(
            f"Too few words ({len(filtered_contributions)}), taking top 50 contributors regardless of threshold"
        )
        sorted_contribs = sorted(
            vocab_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        filtered_contributions = dict(sorted_contribs[:50])
        print(f"Now have {len(filtered_contributions)} words")

    if not filtered_contributions:
        print(f"ERROR: No contributions found for {matrix_name} dimension {dim}!")
        # Create a placeholder image
        placeholder_img = Image.new("RGB", (800, 640), "lightgray")
        draw = ImageDraw.Draw(placeholder_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = (
            f"No contributions\n{matrix_name} Dim {dim}\nActivation: {activation:.6f}"
        )
        draw.text((50, 300), text, fill="black", font=font)

        output_path = os.path.join(
            output_dir, f"{matrix_name.lower()}_dim_{dim:02d}_{word}.png"
        )
        placeholder_img.save(output_path, "PNG")
        return output_path, None

    # This part is new - preparing data for the modal
    dimension_data = {"positive": [], "negative": []}

    # Sort contributions for display
    sorted_filtered_contribs = sorted(
        filtered_contributions.items(), key=lambda item: item[1]
    )

    for word_str, contrib_val in sorted_filtered_contribs:
        idx = stoi.get(word_str, -1)
        if contrib_val >= 0:
            dimension_data["positive"].insert(
                0, {"word": word_str, "value": contrib_val, "index": idx}
            )
        else:
            dimension_data["negative"].append(
                {"word": word_str, "value": contrib_val, "index": idx}
            )

    # Combine all contributions into a single dictionary for word cloud generation
    word_frequencies = {
        word: abs(contrib) for word, contrib in filtered_contributions.items()
    }

    # Define a color function to distinguish positive (black) and negative (red)
    def color_func(word, **kwargs):
        contrib = filtered_contributions.get(word, 0)
        return "black" if contrib >= 0 else "red"

    # Create a single wordcloud for the dimension
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=len(word_frequencies),
        color_func=color_func,
        relative_scaling=0.5,
        min_font_size=6,
    ).generate_from_frequencies(word_frequencies)

    wordcloud_img = wordcloud.to_image()

    # Improved opacity calculation - ensure we always have reasonable visibility
    if abs(activation) < 1e-6:
        opacity = 0.3  # Minimum opacity for very weak activations
        print(f"Very weak activation, using minimum opacity: {opacity}")
    else:
        # Use a more reasonable scale - find the max activation across all dimensions for normalization
        # For now, just use a simple absolute scaling
        opacity = min(1.0, abs(activation) * 10)  # Scale up weak activations
        opacity = max(0.3, opacity)  # Ensure minimum visibility
        print(f"Computed opacity: {opacity}")

    # Apply opacity to the wordcloud image
    if wordcloud_img.mode != "RGBA":
        wordcloud_img = wordcloud_img.convert("RGBA")

    alpha = wordcloud_img.split()[-1]
    alpha = alpha.point(lambda p: int(p * opacity))
    wordcloud_img.putalpha(alpha)

    # Create final image with label and stats
    wordcloud_width, wordcloud_height = wordcloud_img.size
    label_height = 100
    final_width = wordcloud_width
    final_height = wordcloud_height + label_height

    final_img = Image.new("RGB", (final_width, final_height), "white")

    # Paste wordcloud (with opacity) onto the final image
    final_img.paste(wordcloud_img, (0, 0), wordcloud_img)

    # Add labels
    draw = ImageDraw.Draw(final_img)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title = f"{matrix_name} Dimension {dim} - Word: {word}"
    draw.text((10, wordcloud_height + 10), title, fill="black", font=font_large)

    max_abs_contrib = (
        max(abs(c) for c in filtered_contributions.values())
        if filtered_contributions
        else 0
    )

    # Stats
    stats_text = f"Activation: {activation:.4f} | Opacity: {opacity:.2f} | Max contrib: {max_abs_contrib:.4f}"
    draw.text((10, wordcloud_height + 40), stats_text, fill="gray", font=font_small)

    positive_contributions = {w: c for w, c in filtered_contributions.items() if c > 0}
    negative_contributions = {w: c for w, c in filtered_contributions.items() if c < 0}

    # Contribution counts
    contrib_text = f"Pos: {len(positive_contributions)} words | Neg: {len(negative_contributions)} words | Total: {len(filtered_contributions)}/{len(vocab_contributions)}"
    draw.text((10, wordcloud_height + 60), contrib_text, fill="gray", font=font_small)

    # Color indicator for overall activation direction
    color = "green" if activation >= 0 else "red"
    activation_text = (
        f"Overall: {'Positive' if activation >= 0 else 'Negative'} Activation"
    )
    draw.text((10, wordcloud_height + 80), activation_text, fill=color, font=font_small)

    # Save
    output_path = os.path.join(
        output_dir, f"{matrix_name.lower()}_dim_{dim:02d}_{word}.png"
    )
    final_img.save(output_path, "PNG")

    print(f"Saved {matrix_name} dimension {dim} wordcloud: {output_path}")
    print(f"=== END DEBUG {matrix_name} Dimension {dim} ===\n")
    return output_path, dimension_data


def main():
    """Main execution function."""
    # Get configuration
    checkpoint_path = os.environ.get("MODEL")
    probe_text = os.environ.get("PROBE_TEXT", "Knock knock who's there")
    layer_idx = int(os.environ.get("LAYER_INDEX", "0"))

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    # Load model and tokenizer
    token_embeddings, pos_embeddings, model_args, model_state = load_checkpoint(
        checkpoint_path
    )
    stoi, itos = load_tokenizer()
    if not stoi:
        print("Could not load tokenizer. Exiting.")
        sys.exit(1)

    # Load Q, K, V matrices for the specified layer
    W_q, W_k, W_v = load_qkv_matrices(model_state, model_args, layer_idx)
    if W_q is None:
        print("Could not load QKV matrices. Exiting.")
        sys.exit(1)

    # Process the entire probe text
    words = [
        word
        for word in re.findall(r"\\w+|[^\\w\\s]", probe_text.lower(), re.UNICODE)
        if word
    ]

    # Create output directory
    model_name = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join(
        "visualizations", model_name, f"qkv_decomposition_layer_{layer_idx}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # This will store data for all words and all matrices (Q, K, V)
    all_words_data = {}

    for position, word in enumerate(words):
        print(f"\n{'='*20} Processing word: '{word}' (pos {position}) {'='*20}")

        word_repr = get_word_representation(
            word, stoi, token_embeddings, pos_embeddings, position
        )
        if word_repr is None:
            continue

        word_data = {
            "Q": {"activations": [], "dimensions": {}},
            "K": {"activations": [], "dimensions": {}},
            "V": {"activations": [], "dimensions": {}},
        }

        for matrix_name, W_matrix in [("Q", W_q), ("K", W_k), ("V", W_v)]:

            # 1. Compute Q/K/V activations for the current word
            activations = compute_transformed_activations(word_repr, W_matrix)
            word_data[matrix_name]["activations"] = activations.tolist()

            for dim in range(model_args["n_embd"]):
                # 2. Decompose each activation dimension into vocab contributions
                vocab_contributions = compute_vocab_contributions_for_dim(
                    token_embeddings, W_matrix, dim, itos
                )

                # 3. Create wordcloud and get data for the modal
                _, dim_data = create_dimension_wordcloud(
                    vocab_contributions,
                    activations[dim],
                    dim,
                    output_dir,
                    f"{word}_{position}",
                    matrix_name,
                    stoi,
                )
                word_data[matrix_name]["dimensions"][dim] = dim_data

        all_words_data[f"{word}_{position}"] = word_data

    # Generate the comprehensive HTML page for the entire sentence
    generate_html_page(
        output_dir, model_name, words, model_args, all_words_data, layer_idx
    )

    print(f"\nDone! Visualizations saved to: {output_dir}/index.html")


def generate_html_page(
    output_dir, model_name, words, model_args, all_words_data, layer_idx
):
    """Generate HTML page showing Q, K, V decompositions for a sentence."""

    # Convert all data to JSON for embedding in HTML
    import json

    all_data_json = json.dumps(all_words_data)

    # Main table structure for the sentence
    table_html = "<table class='flow-table'>"
    # Header Row
    table_html += "<tr><th>Word (Pos)</th><th>Input (x)</th><th>Query (q)</th><th>Key (k)</th><th>Value (v)</th></tr>"

    # Rows for each word in the sentence
    for position, word in enumerate(words):
        # We need a placeholder for the input 'x' visualization
        x_placeholder = "<div class='placeholder-vis'>Input Repr</div>"

        q_grid = generate_qkv_grid("Q", word, position, all_words_data)
        k_grid = generate_qkv_grid("K", word, position, all_words_data)
        v_grid = generate_qkv_grid("V", word, position, all_words_data)

        table_html += f"""
            <tr>
                <td class='word-label'>'{word}'<br>(pos {position})</td>
                <td>{x_placeholder}</td>
                <td>{q_grid}</td>
                <td>{k_grid}</td>
                <td>{v_grid}</td>
            </tr>
        """

    table_html += "</table>"

    grid_html = ""
    for position, word in enumerate(words):
        grid_html += generate_qkv_grid("Q", word, position, all_words_data)
        grid_html += generate_qkv_grid("K", word, position, all_words_data)
        grid_html += generate_qkv_grid("V", word, position, all_words_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q/K/V Decomposition - {model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
        .header {{ background: linear-gradient(135deg, #4e54c8 0%, #8f94fb 100%); color: white; padding: 30px; text-align: center; }}
        .header h1, .header p {{ margin: 0; }}
        .content {{ padding: 30px; }}
        .tabs {{ display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }}
        .tab-btn {{ padding: 15px 25px; cursor: pointer; background: #f1f1f1; border: none; outline: none; transition: background 0.3s; font-size: 16px; font-weight: bold; }}
        .tab-btn.active {{ background: #fff; border-bottom: 2px solid #4e54c8; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 20px; }}
        .dimension-card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s; cursor: pointer; }}
        .dimension-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .dimension-card img {{ width: 100%; height: 150px; object-fit: cover; }}
        .dimension-card .info {{ padding: 15px; text-align: center; }}
        .dimension-card .info h3 {{ margin: 0 0 5px 0; color: #333; }}
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; padding: 20px; box-sizing: border-box; }}
        .modal-content {{ background: white; border-radius: 8px; max-width: 90%; max-height: 90%; overflow: auto; position: relative; }}
        .modal .close {{ position: absolute; top: 10px; right: 20px; color: #666; font-size: 30px; cursor: pointer; z-index: 1001; }}
        .modal-body {{ padding: 20px; display: flex; gap: 20px; flex-wrap: wrap; }}
        .modal-image {{ flex: 1; min-width: 300px; }}
        .modal-image img {{ width: 100%; border-radius: 4px; }}
        .modal-data {{ flex: 1; min-width: 300px; }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .data-table th {{ background: #4e54c8; color: white; padding: 10px; text-align: left; }}
        .data-table td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
        .data-table .positive {{ color: #000; }}
        .data-table .negative {{ color: #CC0000; }}
        .flow-table th, .flow-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }}
        .flow-table th {{
            background-color: #f2f2f2;
        }}
        .word-label {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .qkv-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 2px;
        }}
        .placeholder-vis {{
            width: 312px; /* 6 * 50px cell + 6 * 2px gap */
            height: 312px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #fafafa;
            border: 1px dashed #ccc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Q/K/V Decomposition Visualizer</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Layer:</strong> {layer_idx}</p>
            <p><strong>Input Text:</strong> "{' '.join(words)}"</p>
        </div>

        <div class="content">
            <div class="description-box">
                <h3>Understanding the Visualization</h3>
                <p>This view decomposes the Query, Key, and Value vectors for each word in the input sentence.
                Each cell in the 6x6 grids represents one of the first 36 dimensions of the Q, K, or V vector.
                The word cloud inside each cell shows which vocabulary words contribute most strongly (positively in black, negatively in red) to the value of that specific dimension.</p>
            </div>

            {table_html}

        </div>
    </div>

    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content">
            <span class="close" onclick="document.getElementById('modal').style.display='none'">&times;</span>
            <div class="modal-body">
                <div class="modal-image">
                    <img id="modal-img" src="">
                </div>
                <div class="modal-data">
                    <h3 id="modal-title"></h3>
                    <h4>Highest Positive Values</h4>
                    <table class="data-table"><tbody id="positive-tbody"></tbody></table>
                    <h4>Most Negative Values</h4>
                    <table class="data-table"><tbody id="negative-tbody"></tbody></table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const allData = {all_data_json};

        function openModal(matrixName, word, position, dimension) {{
            const modal = document.getElementById('modal');
            const modalTitle = document.getElementById('modal-title');
            const modalImg = document.getElementById('modal-img');

            const wordPosKey = `${{word}}_${{position}}`;
            const dimensionData = allData[wordPosKey][matrixName]['dimensions'][dimension];
            const activationValue = allData[wordPosKey][matrixName]['activations'][dimension];

            modalTitle.innerHTML = `${{matrixName}} Dimension ${{dimension}} for '<strong>${{word}}</strong>' (pos ${{position}}) <br> Activation: <span class="${{activationValue >= 0 ? 'positive-val' : 'negative-val'}}">${{activationValue.toFixed(6)}}</span>`;
            modalImg.src = `${{matrixName.toLowerCase()}}_dim_${{String(dimension).padStart(2, '0')}}_${{word}}_${{position}}.png`;

            // Populate data tables (positive and negative)
            loadDimensionData(dimensionData);

            showSection('both'); // Default view
            modal.style.display = 'flex';
        }}

        // The rest of the JS (loadDimensionData, showSection, closeModal) remains largely the same
        // ...
    </script>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML page: {html_path}")


def generate_qkv_grid(matrix_name, word, position, all_words_data):
    """Generates the 6x6 grid of dimension wordmaps for one matrix (Q, K, or V)."""

    n_embd = len(all_words_data[f"{word}_{position}"][matrix_name]["activations"])

    grid_html = (
        f"<div class='qkv-grid' id='{matrix_name.lower()}-{word}-{position}-grid'>"
    )

    # Create a 6x6 grid for the first 36 dimensions
    for dim in range(min(n_embd, 36)):
        img_path = f"{matrix_name.lower()}_dim_{dim:02d}_{word}_{position}.png"

        grid_html += f"""
            <div class="dimension-card" onclick="openModal('{matrix_name}', '{word}', {position}, {dim})">
                <img src="{img_path}" alt="{matrix_name} Dimension {dim} for {word}" loading="lazy">
                <div class="info">
                    <strong>Dim {dim}</strong>
                </div>
            </div>"""

    grid_html += "</div>"
    return grid_html


if __name__ == "__main__":
    main()
