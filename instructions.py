#!/usr/bin/env python3
"""
Simple instructions for using the minimal token visualizer
"""

print("""
🎯 MINIMAL NANOGPT TOKEN VISUALIZER

This is a completely minimal, focused tool that does ONE thing:
Creates word cloud visualizations from NanoGPT token embeddings.

📁 FILES:
- visualize_tokens.py  (main script - does everything)
- requirements.txt     (dependencies)
- test.py             (test with your checkpoint)
- README.md           (basic docs)

🚀 QUICK START:

1. Install dependencies:
   pipenv install
   pipenv shell

2. Run with your checkpoint:
   python visualize_tokens.py /path/to/checkpoint.pt

3. Or test with your existing checkpoint:
   python test.py

📤 OUTPUT:
Creates token_visualizations/ directory with word cloud images.

🔍 DEBUGGING:
The script prints detailed information about:
- Model vocabulary size
- Embedding dimensions  
- Token decoding process
- Sample vocabulary mappings

All in ONE simple, readable Python file!
""")
