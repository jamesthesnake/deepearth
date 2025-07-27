#!/usr/bin/env python3
"""
Diagnose why vision embedding loading is slow
"""

import time
import sys
from pathlib import Path
import os

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache


def time_operations():
    """Time different operations to find bottleneck"""
    
    print("⏱️  Timing Vision Embedding Operations\n")
    
    # Initialize cache
    print("1. Initializing data cache...")
    start = time.time()
    
    original_cwd = os.getcwd()
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    os.chdir(dashboard_dir)
    
    try:
        cache = UnifiedDataCache("dataset_config.json")
        init_time = time.time() - start
        print(f"   ✅ Cache initialization: {init_time:.2f}s")
        
        # Get some observation IDs
        print("\n2. Getting observation IDs...")
        start = time.time()
        obs_ids = get_available_observation_ids(cache, has_vision=True, limit=100)
        get_ids_time = time.time() - start
        print(f"   ✅ Get IDs: {get_ids_time:.2f}s for {len(obs_ids)} IDs")
        
        # Test different batch sizes
        test_sizes = [1, 5, 10, 32]
        
        for batch_size in test_sizes:
            print(f"\n3. Loading {batch_size} embeddings...")
            test_ids = obs_ids[:batch_size]
            
            start = time.time()
            batch_data = get_training_batch(
                cache,
                test_ids,
                include_vision=True,
                include_language=False,
                device='cpu'
            )
            load_time = time.time() - start
            
            time_per_embedding = load_time / batch_size
            print(f"   ✅ Batch size {batch_size}: {load_time:.2f}s total, {time_per_embedding:.2f}s per embedding")
            
            if batch_size == 32:
                # Estimate for full dataset
                total_embeddings = 3518
                estimated_time = (total_embeddings / 32) * load_time
                print(f"\n📊 Estimated time for {total_embeddings} embeddings:")
                print(f"   Sequential: {estimated_time / 60:.1f} minutes")
                print(f"   Parallel (4 workers): {estimated_time / 60 / 4:.1f} minutes")
        
        # Check if it's a memory-mapped file issue
        print("\n4. Checking data source...")
        print(f"   Cache type: {type(cache).__name__}")
        
        # Check if embeddings are being loaded from parquet
        if hasattr(cache, 'vision_loader'):
            print(f"   Vision loader type: {type(cache.vision_loader).__name__}")
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    time_operations()
