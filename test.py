#!/usr/bin/env python3
"""
Test the minimal token visualizer with training data validation
"""

import os
import sys

def test_visualizer():
    """Test the visualizer with your checkpoint and training data."""
    checkpoint_path = "/Users/samueljoseph/Documents/Github/tansaku/nanoGPT/out/knock_6_6_768/ckpt.pt"
    training_data_path = "/Users/samueljoseph/Documents/Github/tansaku/nanoGPT/data/knockknock/knock-knock-jokes-cleaned.txt"
    
    print("Testing minimal token visualizer with training data validation...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training data: {training_data_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
        
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found: {training_data_path}")
        print("Will test without training data validation...")
        training_data_path = None
    
    # Import and run
    try:
        import visualize_tokens
        
        # Override sys.argv for testing
        original_argv = sys.argv
        if training_data_path:
            sys.argv = ['visualize_tokens.py', checkpoint_path, training_data_path]
        else:
            sys.argv = ['visualize_tokens.py', checkpoint_path]
        
        visualize_tokens.main()
        
        # Restore argv
        sys.argv = original_argv
        
        print("✅ Test completed successfully!")
        print("📁 Check token_visualizations/ for results")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_visualizer()
