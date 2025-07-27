#!/usr/bin/env python3
"""
Cache ALL available vision embeddings from the dataset
"""

import pickle
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache


def cache_all_available():
    """Cache all available vision embeddings"""
    
    print("🌍 Caching ALL Available Vision Embeddings\n")
    
    # Change to dashboard directory
    original_cwd = os.getcwd()
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    os.chdir(dashboard_dir)
    
    try:
        # Initialize cache
        print("🔧 Initializing data cache...")
        cache = UnifiedDataCache("dataset_config.json")
        
        # Get ALL available observation IDs
        print("🔍 Getting available observation IDs...")
        all_obs_ids = get_available_observation_ids(
            cache,
            has_vision=True,
            has_language=True,
            limit=None
        )
        
        print(f"✅ Found {len(all_obs_ids)} observations with vision+language")
        
        # Load existing cache if any
        vision_cache = {}
        if os.path.exists('../training/vision_cache_all.pkl'):
            print("\n📂 Loading existing cache...")
            with open('../training/vision_cache_all.pkl', 'rb') as f:
                vision_cache = pickle.load(f)
            print(f"   Found {len(vision_cache)} existing embeddings")
        
        # Filter out already cached
        obs_ids_to_cache = []
        for obs_id in all_obs_ids:
            gbif_id = int(obs_id.split('_')[0])
            if gbif_id not in vision_cache:
                obs_ids_to_cache.append(obs_id)
        
        print(f"📦 Need to cache {len(obs_ids_to_cache)} new embeddings")
        
        if not obs_ids_to_cache:
            print("\n✅ All embeddings already cached!")
            return vision_cache
        
        # Process in batches
        batch_size = 32
        total_time = 0
        failed_count = 0
        
        print(f"\n🚀 Caching {len(obs_ids_to_cache)} embeddings...")
        
        for i in tqdm(range(0, len(obs_ids_to_cache), batch_size), desc="Caching batches"):
            batch_obs_ids = obs_ids_to_cache[i:i + batch_size]
            
            try:
                # Time the batch
                start_time = time.time()
                
                # Load batch
                batch_data = get_training_batch(
                    cache,
                    batch_obs_ids,
                    include_vision=True,
                    include_language=False,
                    device='cpu'
                )
                
                batch_time = time.time() - start_time
                total_time += batch_time
                
                # Store embeddings
                vision_embeddings = batch_data['vision_embeddings'].numpy()
                
                for j, obs_id in enumerate(batch_obs_ids):
                    gbif_id = int(obs_id.split('_')[0])
                    vision_cache[gbif_id] = vision_embeddings[j]
                
                # Save checkpoint every 1000 embeddings
                if len(vision_cache) % 1000 == 0:
                    with open('../training/vision_cache_all_checkpoint.pkl', 'wb') as f:
                        pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"\n💾 Checkpoint saved: {len(vision_cache)} embeddings")
                
                # Show time estimate
                if i > 0:
                    avg_time_per_batch = total_time / ((i + batch_size) / batch_size)
                    remaining_batches = (len(obs_ids_to_cache) - i - batch_size) / batch_size
                    eta_minutes = (avg_time_per_batch * remaining_batches) / 60
                    tqdm.write(f"   ETA: {eta_minutes:.1f} minutes")
                    
            except Exception as e:
                print(f"\n❌ Error in batch: {str(e)}")
                failed_count += len(batch_obs_ids)
                continue
        
        print(f"\n✅ Caching complete!")
        print(f"   Total cached: {len(vision_cache)}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        
        # Save final cache
        os.chdir(original_cwd)
        output_file = 'vision_cache_all.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n💾 Saved to {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024**3:.2f} GB")
        
        return vision_cache
        
    finally:
        os.chdir(original_cwd)


def create_train_test_split(vision_cache):
    """Create train/test splits from all available data"""
    
    print("\n📊 Creating train/test splits...")
    
    # Load observations to get species info
    obs_path = Path("../dashboard/huggingface_dataset/hf_download/observations.parquet")
    import pandas as pd
    obs_df = pd.read_parquet(obs_path)
    
    # Create GBIF to species mapping
    gbif_to_species = {}
    for _, row in obs_df.iterrows():
        gbif_to_species[row['gbif_id']] = row['taxon_name']
    
    # Organize cached embeddings by species
    species_embeddings = {}
    for gbif_id in vision_cache.keys():
        if gbif_id in gbif_to_species:
            species = gbif_to_species[gbif_id]
            if species not in species_embeddings:
                species_embeddings[species] = []
            species_embeddings[species].append(gbif_id)
    
    print(f"   Found {len(species_embeddings)} species with vision embeddings")
    
    # Create train/test splits
    train_cache = {}
    test_cache = {}
    
    for species, gbif_ids in species_embeddings.items():
        gbif_ids = sorted(gbif_ids)  # Ensure consistent ordering
        
        # Use first 80% for train, last 20% for test
        split_idx = int(len(gbif_ids) * 0.8)
        
        train_ids = gbif_ids[:split_idx]
        test_ids = gbif_ids[split_idx:]
        
        for gid in train_ids:
            train_cache[gid] = vision_cache[gid]
        for gid in test_ids:
            test_cache[gid] = vision_cache[gid]
    
    print(f"   Train: {len(train_cache)} embeddings")
    print(f"   Test: {len(test_cache)} embeddings")
    
    # Save splits
    with open('vision_cache_train_all.pkl', 'wb') as f:
        pickle.dump(train_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('vision_cache_test_all.pkl', 'wb') as f:
        pickle.dump(test_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\n✅ Saved train/test splits:")
    print("   - vision_cache_train_all.pkl")
    print("   - vision_cache_test_all.pkl")
    
    return train_cache, test_cache


if __name__ == "__main__":
    # Cache all available embeddings
    vision_cache = cache_all_available()
    
    # Create train/test splits
    if len(vision_cache) > 100:
        train_cache, test_cache = create_train_test_split(vision_cache)
        
        print("\n🎯 Ready for training!")
        print("\nTo use these caches:")
        print("1. cp vision_cache_train_all.pkl vision_cache_train.pkl")
        print("2. cp vision_cache_test_all.pkl vision_cache_test.pkl")
        print("3. python deepearth_corrected_training.py")
