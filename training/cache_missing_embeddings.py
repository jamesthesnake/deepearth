#!/usr/bin/env python3
"""
Complete fix for vision embeddings cache - ensures all training observations have embeddings
"""

import pickle
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache


def get_all_training_observation_ids(max_per_species=20):
    """Get all observation IDs that will be used in training - matching dataset logic exactly"""
    
    print("🔍 Finding all observation IDs used in training...")
    
    # Load observations - matching the exact path from training
    obs_path = Path("../dashboard/huggingface_dataset/hf_download/observations.parquet")
    obs_df = pd.read_parquet(obs_path)
    obs_df = obs_df[obs_df['has_vision'] == True]
    
    # Match the exact logic from DeepEarthDatasetCorrected
    training_data = []
    species_counts = {}
    
    for _, obs in obs_df.iterrows():
        species = obs['taxon_name']
        
        # Initialize count before checking
        if species not in species_counts:
            species_counts[species] = 0
        
        current_count = species_counts[species]
        
        # Train: first max_per_species observations
        if current_count < max_per_species:
            training_data.append({
                'gbif_id': obs['gbif_id'],
                'species': species
            })
        
        # Increment after checking
        species_counts[species] += 1
        
        # Check if we have enough species
        unique_species = set(d['species'] for d in training_data)
        if len(unique_species) >= 10:
            # Check if all species have enough samples
            species_in_data = list(unique_species)
            if all(species_counts.get(s, 0) >= max_per_species for s in species_in_data):
                break
    
    # Extract just the gbif_ids
    training_ids = [d['gbif_id'] for d in training_data]
    
    print(f"✅ Found {len(training_ids)} observations in training set")
    print(f"   From {len(unique_species)} species")
    
    # Show species distribution
    species_dist = {}
    for d in training_data:
        species = d['species']
        species_dist[species] = species_dist.get(species, 0) + 1
    
    print("\n📊 Top 10 species by count:")
    for species, count in sorted(species_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {species}: {count} observations")
    
    return training_ids


def cache_all_vision_embeddings(observation_ids, output_file='vision_cache_train_complete.pkl'):
    """Cache vision embeddings for all provided observation IDs"""
    
    print(f"\n📦 Caching vision embeddings for {len(observation_ids)} observations...")
    
    # Check existing cache
    existing_cache = {}
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            existing_cache = pickle.load(f)
        print(f"   Existing cache has {len(existing_cache)} embeddings")
    
    # Find what needs to be cached
    missing_ids = [obs_id for obs_id in observation_ids if obs_id not in existing_cache]
    print(f"   Need to cache {len(missing_ids)} new embeddings")
    
    if not missing_ids:
        print("✅ All embeddings already cached!")
        return existing_cache
    
    # Initialize data cache
    original_cwd = os.getcwd()
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    os.chdir(dashboard_dir)
    
    try:
        cache = UnifiedDataCache("dataset_config.json")
        
        # Process in batches
        batch_size = 32
        vision_cache = existing_cache.copy()
        failed_ids = []
        
        for i in tqdm(range(0, len(missing_ids), batch_size), desc="Loading vision embeddings"):
            batch_ids = missing_ids[i:i + batch_size]
            
            try:
                # Load vision embeddings
                batch_data = get_training_batch(
                    cache,
                    batch_ids,
                    include_vision=True,
                    include_language=False,
                    device='cpu'
                )
                
                # Extract and store embeddings
                vision_embeddings = batch_data['vision_embeddings']
                
                for j, obs_id in enumerate(batch_ids):
                    embedding = vision_embeddings[j].numpy()
                    
                    # Validate shape
                    if embedding.shape == (8, 24, 24, 1408):
                        vision_cache[obs_id] = embedding
                    else:
                        print(f"\n⚠️  Unexpected shape for {obs_id}: {embedding.shape}")
                        failed_ids.append(obs_id)
                        
            except Exception as e:
                print(f"\n⚠️  Error loading batch starting at {batch_ids[0]}: {str(e)}")
                failed_ids.extend(batch_ids)
                continue
        
        # Report results
        print(f"\n📊 Caching complete:")
        print(f"   Successfully cached: {len(vision_cache)} embeddings")
        print(f"   Failed: {len(failed_ids)} embeddings")
        
        if failed_ids:
            print(f"\n⚠️  Failed observation IDs:")
            for obs_id in failed_ids[:10]:
                print(f"   - {obs_id}")
            if len(failed_ids) > 10:
                print(f"   ... and {len(failed_ids) - 10} more")
        
        # Save cache
        print(f"\n💾 Saving cache to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✅ Cache saved successfully!")
        print(f"   Total embeddings: {len(vision_cache)}")
        print(f"   File size: {os.path.getsize(output_file) / 1024**3:.2f} GB")
        
        return vision_cache
        
    finally:
        os.chdir(original_cwd)


def verify_cache_completeness(cache_file, expected_ids):
    """Verify that cache contains all expected observation IDs"""
    
    print(f"\n🔍 Verifying cache completeness...")
    
    with open(cache_file, 'rb') as f:
        vision_cache = pickle.load(f)
    
    missing = [obs_id for obs_id in expected_ids if obs_id not in vision_cache]
    
    print(f"   Cache has {len(vision_cache)} embeddings")
    print(f"   Expected {len(expected_ids)} embeddings")
    print(f"   Missing {len(missing)} embeddings")
    
    if missing:
        print(f"\n⚠️  Still missing {len(missing)} embeddings:")
        for obs_id in missing[:10]:
            print(f"   - {obs_id}")
    else:
        print("✅ All expected embeddings are cached!")
    
    return missing


def main():
    """Main execution"""
    
    print("🌍 DeepEarth Complete Vision Cache Fix\n")
    
    # Step 1: Get all observation IDs that training will use
    training_ids = get_all_training_observation_ids(max_per_species=20)
    
    # Step 2: Cache all vision embeddings
    cache_all_vision_embeddings(training_ids, 'vision_cache_train_complete.pkl')
    
    # Step 3: Verify completeness
    missing = verify_cache_completeness('vision_cache_train_complete.pkl', training_ids)
    
    # Step 4: Create backup and replace original
    if not missing:
        print("\n🔄 Replacing original cache...")
        
        # Backup original
        if os.path.exists('vision_cache_train.pkl'):
            os.rename('vision_cache_train.pkl', 'vision_cache_train_backup.pkl')
            print("   Backed up original to vision_cache_train_backup.pkl")
        
        # Replace with complete cache
        os.rename('vision_cache_train_complete.pkl', 'vision_cache_train.pkl')
        print("✅ Replaced vision_cache_train.pkl with complete cache")
    
    print("\n✨ Done! Your training should now run without missing embeddings.")


if __name__ == "__main__":
    main()
