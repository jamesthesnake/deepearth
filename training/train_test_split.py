#!/usr/bin/env python3
"""
Use the official DeepEarth train/test split configuration
This ensures consistency with the intended dataset usage
"""

import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import os
import sys
from tqdm import tqdm

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch
from data_cache import UnifiedDataCache


def load_split_config(config_path='config/central_florida_split.json'):
    """Load the official train/test split configuration"""
    
    print(f"📋 Loading split configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract observation mappings
    observation_mappings = config['observation_mappings']
    
    # Separate by split
    train_obs = []
    test_obs = []
    
    for obs_id, metadata in observation_mappings.items():
        if metadata['split'] == 'train':
            train_obs.append({
                'obs_id': obs_id,
                'gbif_id': metadata['gbif_id'],
                'species': metadata['taxon_name']  # Use taxon_name
            })
        else:  # test
            test_obs.append({
                'obs_id': obs_id,
                'gbif_id': metadata['gbif_id'],
                'species': metadata['taxon_name']  # Use taxon_name
            })
    
    print(f"✅ Found {len(train_obs)} train and {len(test_obs)} test observations")
    
    # Show species distribution
    train_species = set(obs['species'] for obs in train_obs)
    test_species = set(obs['species'] for obs in test_obs)
    
    print(f"   Train species: {len(train_species)}")
    print(f"   Test species: {len(test_species)}")
    print(f"   Common species: {len(train_species & test_species)}")
    
    return train_obs, test_obs, config


def cache_split_embeddings(train_obs, test_obs, use_existing=True):
    """Cache embeddings for the official split"""
    
    print("\n🔍 Checking existing caches...")
    
    # Load existing caches if available
    train_cache = {}
    test_cache = {}
    
    if use_existing and os.path.exists('vision_cache_train.pkl'):
        with open('vision_cache_train.pkl', 'rb') as f:
            existing = pickle.load(f)
        print(f"   Found {len(existing)} embeddings in vision_cache_train.pkl")
        
        # Convert to use GBIF IDs as keys
        for key, value in existing.items():
            if isinstance(key, int):
                train_cache[key] = value
    
    if use_existing and os.path.exists('vision_cache_test.pkl'):
        with open('vision_cache_test.pkl', 'rb') as f:
            existing = pickle.load(f)
        print(f"   Found {len(existing)} embeddings in vision_cache_test.pkl")
        
        for key, value in existing.items():
            if isinstance(key, int):
                test_cache[key] = value
    
    # Find what needs to be cached
    train_to_cache = [obs for obs in train_obs if obs['gbif_id'] not in train_cache]
    test_to_cache = [obs for obs in test_obs if obs['gbif_id'] not in test_cache]
    
    print(f"\n📦 Need to cache:")
    print(f"   Train: {len(train_to_cache)} embeddings")
    print(f"   Test: {len(test_to_cache)} embeddings")
    
    if len(train_to_cache) + len(test_to_cache) == 0:
        print("\n✅ All embeddings already cached!")
        return train_cache, test_cache
    
    # Initialize data cache
    print("\n🔧 Initializing data cache...")
    original_cwd = os.getcwd()
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    os.chdir(dashboard_dir)
    
    try:
        cache = UnifiedDataCache("dataset_config.json")
        
        # Cache train embeddings
        if train_to_cache:
            print(f"\n📥 Caching {len(train_to_cache)} train embeddings...")
            train_cache = cache_observations(train_to_cache, cache, train_cache)
        
        # Cache test embeddings  
        if test_to_cache:
            print(f"\n📥 Caching {len(test_to_cache)} test embeddings...")
            test_cache = cache_observations(test_to_cache, cache, test_cache)
        
    finally:
        os.chdir(original_cwd)
    
    # Save updated caches
    print("\n💾 Saving caches...")
    
    with open('vision_cache_train_split.pkl', 'wb') as f:
        pickle.dump(train_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved {len(train_cache)} train embeddings")
    
    with open('vision_cache_test_split.pkl', 'wb') as f:
        pickle.dump(test_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved {len(test_cache)} test embeddings")
    
    return train_cache, test_cache


def cache_observations(obs_list, data_cache, existing_cache):
    """Cache embeddings for a list of observations"""
    
    batch_size = 32
    
    for i in tqdm(range(0, len(obs_list), batch_size)):
        batch = obs_list[i:i + batch_size]
        obs_ids = [obs['obs_id'] for obs in batch]
        
        try:
            # Load batch
            batch_data = get_training_batch(
                data_cache,
                obs_ids,
                include_vision=True,
                include_language=False,
                device='cpu'
            )
            
            # Store embeddings
            vision_embeddings = batch_data['vision_embeddings'].numpy()
            
            for j, obs in enumerate(batch):
                existing_cache[obs['gbif_id']] = vision_embeddings[j]
                
        except Exception as e:
            print(f"\n⚠️  Error loading batch: {e}")
            continue
    
    return existing_cache


class DeepEarthSplitDataset(Dataset):
    """Dataset using the official split configuration"""
    
    def __init__(self, observations, vision_cache, language_cache=None):
        self.observations = observations
        self.vision_cache = vision_cache
        self.language_cache = language_cache or {}
        
        # Filter to only observations with cached embeddings
        self.data = []
        for obs in observations:
            if obs['gbif_id'] in vision_cache:
                self.data.append(obs)
        
        print(f"   Using {len(self.data)} / {len(observations)} observations (with cached embeddings)")
        
        # Create species mapping
        all_species = sorted(list(set(obs['species'] for obs in self.data)))
        self.species_to_idx = {s: i for i, s in enumerate(all_species)}
        self.num_species = len(all_species)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs = self.data[idx]
        gbif_id = obs['gbif_id']
        
        # Get vision embedding
        vision_emb = self.vision_cache[gbif_id]
        
        # Get or create language embedding
        if gbif_id in self.language_cache:
            lang_emb = self.language_cache[gbif_id]
        else:
            # Create a dummy language embedding for now
            lang_emb = np.random.randn(7168).astype(np.float32)
        
        return {
            'vision_embedding': torch.tensor(vision_emb, dtype=torch.float32),
            'language_embedding': torch.tensor(lang_emb, dtype=torch.float32),
            'species': obs['species'],
            'species_idx': self.species_to_idx[obs['species']],
            'gbif_id': gbif_id
        }


def main():
    print("🌍 DeepEarth Training with Official Split Configuration\n")
    
    # Load split configuration
    train_obs, test_obs, config = load_split_config()
    
    # Try to use existing caches first
    print("\n🔍 Checking what embeddings we already have...")
    
    # Quick check - just see what's available without caching new ones
    train_cache = {}
    test_cache = {}
    
    if os.path.exists('vision_cache_train.pkl'):
        with open('vision_cache_train.pkl', 'rb') as f:
            existing = pickle.load(f)
        
        # Match against split
        for obs in train_obs:
            if obs['gbif_id'] in existing:
                train_cache[obs['gbif_id']] = existing[obs['gbif_id']]
    
    if os.path.exists('vision_cache_test.pkl'):
        with open('vision_cache_test.pkl', 'rb') as f:
            existing = pickle.load(f)
        
        for obs in test_obs:
            if obs['gbif_id'] in existing:
                test_cache[obs['gbif_id']] = existing[obs['gbif_id']]
    
    print(f"\n📊 Available embeddings from existing caches:")
    print(f"   Train: {len(train_cache)} / {len(train_obs)}")
    print(f"   Test: {len(test_cache)} / {len(test_obs)}")
    
    if len(train_cache) < 50 or len(test_cache) < 10:
        print("\n⚠️  Not enough cached embeddings for training!")
        print("   You need to cache more embeddings first.")
        
        # Ask if they want to cache some
        response = input("\nCache 100 embeddings to get started? (y/n): ")
        if response.lower() == 'y':
            # Cache a small subset
            train_obs_subset = train_obs[:80]
            test_obs_subset = test_obs[:20]
            train_cache, test_cache = cache_split_embeddings(
                train_obs_subset, 
                test_obs_subset,
                use_existing=False
            )
        else:
            return
    
    # Create datasets
    print("\n📊 Creating datasets...")
    train_dataset = DeepEarthSplitDataset(train_obs, train_cache)
    test_dataset = DeepEarthSplitDataset(test_obs, test_cache)
    
    print(f"\n🎯 Final dataset sizes:")
    print(f"   Train: {len(train_dataset)} observations")
    print(f"   Test: {len(test_dataset)} observations")
    print(f"   Species: {train_dataset.num_species}")
    
    if len(train_dataset) < 10 or len(test_dataset) < 2:
        print("\n❌ Not enough data for training!")
        return
    
    # Create dataloaders
    batch_size = min(32, len(train_dataset) // 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Import and train model
    from deepearth_crossmodal_aligned import MLPUNetCorrected, train_corrected
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPUNetCorrected(num_species=train_dataset.num_species).to(device)
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    metrics = train_corrected(
        model,
        train_loader,
        test_loader,
        epochs=20,
        device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_species': train_dataset.num_species,
        'species_to_idx': train_dataset.species_to_idx,
        'metrics': metrics,
        'split_config': config['metadata']
    }, 'deepearth_split_model.pth')
    
    print("\n✅ Training complete!")
    print("   Model saved to deepearth_split_model.pth")


if __name__ == "__main__":
    main()
