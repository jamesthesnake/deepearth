#!/usr/bin/env python3
"""
Simple robust cacher that saves after every batch
"""

import pickle
import pandas as pd
from pathlib import Path
import sys
import os
import time
from tqdm import tqdm
import json

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache


def save_checkpoint(cache, checkpoint_file='vision_cache_checkpoint.pkl'):
    """Save checkpoint"""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(checkpoint_file='vision_cache_checkpoint.pkl'):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return {}


def save_progress_json(progress, filename='cache_progress.json'):
    """Save progress info"""
    with open(filename, 'w') as f:
        json.dump(progress, f, indent=2)


def simple_cache_all():
    """Simple caching with saves after every batch"""
    
    print("🌍 Simple Robust Vision Cacher\n")
    
    # Load any existing checkpoint
    vision_cache = load_checkpoint()
    if vision_cache:
        print(f"📂 Loaded checkpoint with {len(vision_cache)} embeddings")
    
    # Get required GBIF IDs
    print("\n📋 Getting required GBIF IDs...")
    obs_path = Path("../dashboard/huggingface_dataset/hf_download/observations.parquet")
    obs_df = pd.read_parquet(obs_path)
    obs_df = obs_df[obs_df['has_vision'] == True]
    
    # Get training data
    training_data = []
    species_counts = {}
    
    for _, obs in obs_df.iterrows():
        species = obs['taxon_name']
        
        if species not in species_counts:
            species_counts[species] = 0
        
        if species_counts[species] < 20:
            training_data.append({
                'gbif_id': obs['gbif_id'],
                'species': species
            })
        
        species_counts[species] += 1
        
        if len(set(d['species'] for d in training_data)) >= 10:
            if all(species_counts.get(s, 0) >= 20 for s in set(d['species'] for d in training_data)):
                break
    
    required_gbif_ids = [d['gbif_id'] for d in training_data]
    print(f"✅ Need {len(required_gbif_ids)} total embeddings")
    
    # Filter out already cached
    remaining_gbif_ids = [gid for gid in required_gbif_ids if gid not in vision_cache]
    print(f"📦 Need to cache {len(remaining_gbif_ids)} new embeddings")
    
    if not remaining_gbif_ids:
        print("\n✅ All embeddings already cached!")
        return vision_cache
    
    # Initialize data cache
    print("\n🔧 Initializing data cache...")
    original_cwd = os.getcwd()
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    os.chdir(dashboard_dir)
    
    try:
        cache = UnifiedDataCache("dataset_config.json")
        
        # Get observation ID mapping
        print("🔍 Getting observation IDs...")
        all_obs_ids = get_available_observation_ids(
            cache,
            has_vision=True,
            has_language=True,
            limit=None
        )
        
        # Create mapping
        gbif_to_obs = {}
        for obs_id in all_obs_ids:
            gbif_id = obs_id.split('_')[0]
            if gbif_id not in gbif_to_obs:
                gbif_to_obs[gbif_id] = []
            gbif_to_obs[gbif_id].append(obs_id)
        
        # Map remaining GBIF IDs
        to_cache = []
        unmapped = []
        
        for gbif_id in remaining_gbif_ids:
            gbif_str = str(gbif_id)
            if gbif_str in gbif_to_obs:
                to_cache.append((gbif_id, gbif_to_obs[gbif_str][0]))
            else:
                unmapped.append(gbif_id)
        
        print(f"✅ Mapped {len(to_cache)} GBIF IDs")
        if unmapped:
            print(f"⚠️  {len(unmapped)} GBIF IDs have no vision data")
            # Save unmapped IDs for debugging
            with open('unmapped_gbif_ids.txt', 'w') as f:
                for gid in unmapped:
                    f.write(f"{gid}\n")
        
        # Process in small batches with saves
        batch_size = 16  # Smaller batches
        failed_count = 0
        
        print(f"\n🚀 Caching {len(to_cache)} embeddings...")
        
        for i in tqdm(range(0, len(to_cache), batch_size), desc="Caching"):
            batch = to_cache[i:i + batch_size]
            gbif_ids = [g for g, _ in batch]
            obs_ids = [o for _, o in batch]
            
            try:
                # Load batch
                start_time = time.time()
                batch_data = get_training_batch(
                    cache,
                    obs_ids,
                    include_vision=True,
                    include_language=False,
                    device='cpu'
                )
                load_time = time.time() - start_time
                
                # Store embeddings
                vision_embeddings = batch_data['vision_embeddings'].numpy()
                
                for j, gbif_id in enumerate(gbif_ids):
                    vision_cache[gbif_id] = vision_embeddings[j]
                
                # Save checkpoint after EVERY batch
                save_checkpoint(vision_cache)
                
                # Save progress info
                progress = {
                    'total_cached': len(vision_cache),
                    'last_batch': i + len(batch),
                    'total_needed': len(required_gbif_ids),
                    'batch_time': load_time,
                    'failed': failed_count
                }
                save_progress_json(progress)
                
            except Exception as e:
                print(f"\n❌ Error in batch {i//batch_size}: {str(e)}")
                failed_count += len(batch)
                
                # Save failed IDs
                with open('failed_batch_ids.txt', 'a') as f:
                    f.write(f"Batch {i//batch_size}: {gbif_ids}\n")
                    f.write(f"Error: {str(e)}\n\n")
                
                continue
        
        print(f"\n✅ Caching complete!")
        print(f"   Total cached: {len(vision_cache)}")
        print(f"   Failed: {failed_count}")
        
        # Save final cache
        output_file = 'vision_cache_train_complete.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"💾 Saved to {output_file}")
        print(f"   Size: {os.path.getsize(output_file) / 1024**3:.2f} GB")
        
        # Clean up checkpoint
        if os.path.exists('vision_cache_checkpoint.pkl'):
            os.remove('vision_cache_checkpoint.pkl')
        if os.path.exists('cache_progress.json'):
            os.remove('cache_progress.json')
        
        return vision_cache
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Run the simple cacher
    vision_cache = simple_cache_all()
    
    # Check specific IDs
    check_ids = [3044830298, 3391675877, 3858956662, 4903847224, 4161982655]
    print("\n🔍 Checking previously missing IDs:")
    for gbif_id in check_ids:
        if gbif_id in vision_cache:
            print(f"   ✅ {gbif_id} now cached")
        else:
            print(f"   ❌ {gbif_id} still missing")
    
    # If we have enough embeddings, replace the original
    if len(vision_cache) > 3000:  # 85% of expected
        print("\n✅ Have enough embeddings to proceed with training!")
        if input("Replace vision_cache_train.pkl? (y/n): ").lower() == 'y':
            os.rename('vision_cache_train.pkl', 'vision_cache_train_old.pkl')
            os.rename('vision_cache_train_complete.pkl', 'vision_cache_train.pkl')
            print("✅ Replaced vision_cache_train.pkl")
