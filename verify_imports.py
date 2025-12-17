#!/usr/bin/env python3
"""Verification script to test that all imports work after restructuring."""

import sys

def test_dataset_imports():
    """Test that dataset module can be imported."""
    try:
        from dataset import get_dataset_flickr, textprocess, textprocess_train, create_dataset
        print("✓ Dataset module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Dataset module imports failed: {e}")
        return False

def test_utils_imports():
    """Test that utils module can be imported."""
    try:
        from utils.networks import CLIPModel_full
        from utils.vl_distill_utils import load_or_process_file
        from utils.epoch import epoch, epoch_test, itm_eval, epoch_test_cc3m
        print("✓ Utils module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Utils module imports failed: {e}")
        return False

def test_evaluation_imports():
    """Test that evaluation module can import dataset and utils."""
    try:
        sys.path.insert(0, 'evaluation')
        sys.path.append('..')
        from dataset import get_dataset_flickr
        from utils.networks import CLIPModel_full
        print("✓ Evaluation can import dataset and utils modules")
        return True
    except ImportError as e:
        print(f"✗ Evaluation cannot import modules: {e}")
        return False

if __name__ == "__main__":
    print("Verifying imports after restructuring...\n")
    
    success = True
    success = test_dataset_imports() and success
    success = test_utils_imports() and success
    success = test_evaluation_imports() and success
    
    if success:
        print("\n✓ All imports verified successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some imports failed. Please check the errors above.")
        sys.exit(1)
