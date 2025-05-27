#!/usr/bin/env python3
"""
Test the minimal token visualizer
"""

import os
import sys

def test_visualizer():
    """Test the visualizer with your checkpoint."""
    checkpoint_path = "/Users/samueljoseph/Documents/Github/karpathy/nanoGPT/out/knock_distilgpt2_restart/ckpt-best.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    print("Testing minimal token visualizer...")
    
    # Import and run
    try:
        import visualize_tokens
        
        # Override sys.argv for testing
        original_argv = sys.argv
        sys.argv = ['visualize_tokens.py', checkpoint_path]
        
        visualize_tokens.main()
        
        # Restore argv
        sys.argv = original_argv
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_visualizer()
