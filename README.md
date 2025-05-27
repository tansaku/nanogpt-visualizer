# NanoGPT Token Visualizer (Minimal)

A minimal tool to visualize token embeddings from NanoGPT checkpoint files with training data validation.

## What it does

1. Loads a NanoGPT `.pt` checkpoint file
2. Extracts token embeddings
3. Creates word cloud visualizations for embedding dimensions
4. **NEW**: Validates which tokens are from training data vs. inherited from GPT-2

## Installation

```bash
pipenv install
pipenv shell
```

## Usage

**Basic usage:**
```bash
python visualize_tokens.py /path/to/checkpoint.pt
```

**With training data validation:**
```bash
python visualize_tokens.py /path/to/checkpoint.pt /path/to/training_data.txt
```

## Output

Creates word cloud images showing:
- **Black tokens**: Positive values (from training data if provided)
- **Red tokens**: Negative values (from training data if provided)  
- **Gray tokens**: Positive values NOT from training data
- **Pink tokens**: Negative values NOT from training data

## Understanding the Results

If you see gray/pink tokens (non-training), it suggests:
1. Your model was fine-tuned from GPT-2 (retaining original embeddings)
2. OR your model uses full GPT-2 vocabulary but was trained from scratch
3. The gray/pink tokens show GPT-2's influence on your model
