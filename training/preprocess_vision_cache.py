#!/usr/bin/env python3
"""
Preprocess vision embeddings by pre-pooling them to reduce memory usage
from ~26MB per sample to ~2.8KB per sample
"""

import pickle
import numpy as np
from tqdm import tqdm
import os

def preprocess_vision_cache(input_file, output_file):
    """Convert 5D vision tensors to pre-pooled 1D vectors"""
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print(f"Loading original vision cache from {input_file}...")
    with open(input_file, 'rb') as f:
        raw = pickle.load(f)
    
    print(f"Loaded {len(raw)} embeddings")
    
    # Check the shape of the first embedding
    sample_key = list(raw.keys())[0]
    sample_shape = raw[sample_key].shape
    print(f"Original embedding shape: {sample_shape}")
    
    # Pre-pool and convert to float16
    print("Pre-pooling embeddings...")
    pooled = {}
    
    for k, v in tqdm(raw.items(), desc="Pooling"):
        # v has shape (8, 24, 24, 1408)
        # Pool over temporal and spatial dimensions
        pooled_vec = v.mean(axis=(0, 1, 2))  # Shape: (1408,)
        pooled[k] = pooled_vec.astype(np.float16)  # Save memory with float16
    
    # Save the pooled cache
    print(f"Saving pooled cache to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(pooled, f)
    
    # Calculate size reduction
    original_size_mb = len(raw) * 8 * 24 * 24 * 1408 * 4 / (1024 * 1024)
    new_size_mb = len(pooled) * 1408 * 2 / (1024 * 1024)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Original size: ~{original_size_mb:.1f} MB")
    print(f"New size: ~{new_size_mb:.1f} MB")
    print(f"Reduction: {(1 - new_size_mb/original_size_mb)*100:.1f}%")

if __name__ == "__main__":
    # Process training cache
    print("Processing training vision cache...")
    preprocess_vision_cache('vision_cache_train.pkl', 'vision_cache_train_pooled.pkl')
    
    print("\n" + "="*50 + "\n")
    
    # Process test cache
    print("Processing test vision cache...")
    preprocess_vision_cache('vision_cache_test.pkl', 'vision_cache_test_pooled.pkl')
