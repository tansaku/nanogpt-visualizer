# NanoGPT Transformer Visualizer

This repository contains a suite of Python scripts to generate in-depth visualizations of the internal workings of a NanoGPT model. It is designed to trace the flow of information step-by-step through the transformer architecture, from the initial embeddings to the final prediction.

The primary output is a detailed HTML page (`index.html`) that shows the state of the model's representations at every stage of processing for a given input sentence.

[![Visualization Screenshot](screenshot.png)](https://tansaku.github.io/nanogpt-visualizer/)

## Features

- **Full Model Flow:** Visualize the transformation of token representations through every layer of the transformer.
- **Input Analysis:** See the difference between a token's raw embedding and its position-aware combined embedding.
- **Attention Deep Dive:** Inspect attention patterns, Q, K, and V vectors, and their compositions.
- **Final Prediction Breakdown:** Analyze the model's final output, including the dot product breakdown that determines the next token prediction.
- **Interactive HTML Report:** An easy-to-navigate HTML page with collapsible sections, tooltips, and pop-up modals with more details.
- **GitHub Pages Deployment:** Automatically deploy your generated visualizations to a live website.

## How to Use

### 1. Prerequisites

- A working installation of Python 3.9 or later.
- `pip`, the Python package installer.

### 2. Setup

First, clone this repository:
```bash
git clone https://github.com/tansaku/nanogpt-visualizer.git
cd nanogpt-visualizer
```

Next, install Pipenv, which is used to manage the project's dependencies in a virtual environment:
```bash
pip install pipenv
```

Install the required dependencies using Pipenv:
```bash
pipenv install
pipenv shell
```

### 3. Configure Your Environment

The scripts in this repository use a `.env` file to manage configuration, such as model paths and the sentence to analyze.

1.  **Create your environment file.** Copy the provided template to a new `.env` file:
    ```bash
    cp env.template .env
    ```

2.  **Edit your `.env` file.** Open the `.env` file and update the variables to point to your model files and the desired probe sentence. The model files (`ckpt.pt`, `meta.pkl`) can be located anywhere on your computer.

    *Example `.env` configuration:*
    ```
    # Path to the NanoGPT model checkpoint file.
    MODEL="/path/to/your/nanogpt/out/my_model/ckpt.pt"

    # Path to the tokenizer metadata file (meta.pkl).
    NANOGPT_META_PATH="/path/to/your/nanogpt/data/my_dataset/meta.pkl"

    # The sentence you want to generate a visualization for.
    PROBE_SENTENCE="the quick brown fox"
    ```

### 4. Generate the Visualization

Once your `.env` file is configured, you can run the main visualization script. The script will automatically load the variables from your `.env` file.

```bash
python visualize_attention_flow_wordmaps.py
```

The script will generate a comprehensive set of images and an `index.html` file inside a new directory within `visualizations/`. The output directory is named after your model checkpoint file.

### 5. View and Deploy

You can open the generated `index.html` file (e.g., `visualizations/my_model_name/full_model_flow/index.html`) in your local browser to view the interactive report.

This repository is configured with a GitHub Action to automatically deploy the contents of a specified directory to GitHub Pages. To use this for your own model, you will need to:

1.  Commit and push all the generated files from your model's output directory.
2.  Update the path in `.github/workflows/deploy.yml` to point to that directory.
3.  The visualization will be available at `https://<your-username>.github.io/nanogpt-visualizer/`.

## Included Visualization Scripts

This repository contains several scripts for different kinds of analysis:

-   `visualize_attention_flow_wordmaps.py`: The main script for the full, end-to-end model visualization.
-   `visualize_tokens.py`: Generates word clouds for each dimension of the token embedding space.
-   `visualize_positions.py`: Generates word clouds for each dimension of the positional embedding space.
-   `visualize_qkv_decomposition.py`: Decomposes Q, K, and V transformations.
-   ... and others for more specific deep dives.

Each script can be explored for more targeted analysis of the model's components.

## Further Context

This repository was developed to support the "Transformers, From the Inside Out" talk. You can find the slides and a recording of the presentation here:

-   **[Presentation Slides (PDF)](TransformersInsideOut-part2.pdf)**
-   **Presentation Video:**

    [![Transformers, From the Inside Out](https://img.youtube.com/vi/Y1Iv5bH3xXo/0.jpg)](https://youtu.be/Y1Iv5bH3xXo)
