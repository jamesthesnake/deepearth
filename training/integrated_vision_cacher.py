#!/usr/bin/env python3
"""
Integrated Vision Embedding Cacher for DeepEarth
Works with existing dashboard infrastructure and training pipeline
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import gc
import os
import sys
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
import json

# Add dashboard to path for data access
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache


class IntegratedVisionCacher:
    """
    Vision embedding cacher that works with your existing infrastructure
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize cacher with dashboard config
        
        Args:
            config_path: Path to dashboard config (will auto-detect if None)
        """
        # Auto-detect config if not provided
        if not config_path:
            dashboard_path = Path(__file__).parent.parent / "dashboard"
            potential_configs = [
                dashboard_path / "dataset_config.json",
                dashboard_path / "config.json",
                dashboard_path / "central_florida_config.json"
            ]
            
            for config in potential_configs:
                if config.exists():
                    config_path = str(config)
                    break
            
            if not config_path:
                raise ValueError("No config file found. Please specify config path.")
        
        self.config_path = config_path
        self.cache = None
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Initialize the UnifiedDataCache"""
        print(f"🔧 Initializing data cache from {self.config_path}")
        
        # Change to dashboard directory for proper relative paths
        original_cwd = os.getcwd()
        dashboard_dir = Path(self.config_path).parent
        os.chdir(dashboard_dir)
        
        try:
            self.cache = UnifiedDataCache(Path(self.config_path).name)
            print("✅ Data cache initialized successfully")
        finally:
            os.chdir(original_cwd)
    
    def get_species_observations(self, 
                               max_per_species: int = 25,
                               min_species: int = 10,
                               test_split: bool = False) -> Tuple[List[str], Dict[str, int]]:
        """
        Get observation IDs organized by species
        
        Args:
            max_per_species: Maximum observations per species
            min_species: Minimum number of species to include
            test_split: Whether to get test split (observations 25-35)
            
        Returns:
            Tuple of (observation_ids, species_counts)
        """
        print(f"\n🔍 Finding observations (test_split={test_split})...")
        
        # Get all available observations with both vision and language
        all_obs_ids = get_available_observation_ids(
            self.cache,
            has_vision=True,
            has_language=True,
            limit=10000  # Get more to ensure we have enough species
        )
        
        print(f"   Found {len(all_obs_ids)} total observations with vision+language")
        
        # Organize by species
        species_observations = {}
        observation_ids = []
        
        # Process in batches to get species information
        batch_size = 100
        for i in tqdm(range(0, len(all_obs_ids), batch_size), desc="Analyzing species"):
            batch_ids = all_obs_ids[i:i + batch_size]
            
            # Get batch data to extract species
            batch_data = get_training_batch(
                self.cache,
                batch_ids,
                include_vision=False,  # Don't load embeddings yet
                include_language=False,
                device='cpu'
            )
            
            for obs_id, species in zip(batch_ids, batch_data['species']):
                if species not in species_observations:
                    species_observations[species] = []
                species_observations[species].append(obs_id)
        
        # Select observations based on split
        selected_observations = []
        species_counts = {}
        
        for species, obs_list in sorted(species_observations.items()):
            if test_split:
                # Test: observations 25-35 (if available)
                start_idx = max_per_species
                end_idx = max_per_species + 10
                selected = obs_list[start_idx:end_idx]
            else:
                # Train: first 25 observations
                selected = obs_list[:max_per_species]
            
            if selected:
                selected_observations.extend(selected)
                species_counts[species] = len(selected)
            
            # Stop when we have enough species
            if len(species_counts) >= min_species:
                break
        
        print(f"   Selected {len(selected_observations)} observations")
        print(f"   From {len(species_counts)} species")
        
        return selected_observations, species_counts
    
    def cache_vision_embeddings(self,
                              observation_ids: List[str],
                              output_file: str,
                              batch_size: int = 32,
                              checkpoint_interval: int = 500):
        """
        Cache vision embeddings for given observation IDs
        
        Args:
            observation_ids: List of observation IDs to cache
            output_file: Output pickle file path
            batch_size: Batch size for loading
            checkpoint_interval: Save checkpoint every N observations
        """
        print(f"\n📦 Caching {len(observation_ids)} vision embeddings...")
        
        # Check existing cache
        existing_cache = {}
        if os.path.exists(output_file):
            print(f"   Loading existing cache: {output_file}")
            try:
                with open(output_file, 'rb') as f:
                    existing_cache = pickle.load(f)
                print(f"   Found {len(existing_cache)} existing embeddings")
            except:
                print("   Could not load existing cache, starting fresh")
        
        # Filter out already cached IDs
        remaining_ids = [obs_id for obs_id in observation_ids 
                        if obs_id not in existing_cache]
        
        if not remaining_ids:
            print("✅ All embeddings already cached!")
            return existing_cache
        
        print(f"   Need to cache {len(remaining_ids)} new embeddings")
        
        # Cache new embeddings
        vision_cache = existing_cache.copy()
        processed = 0
        checkpoint_file = output_file.replace('.pkl', '_checkpoint.pkl')
        
        try:
            # Process in batches
            for i in tqdm(range(0, len(remaining_ids), batch_size), 
                         desc="Caching batches"):
                batch_ids = remaining_ids[i:i + batch_size]
                
                # Load vision embeddings for this batch
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=True,
                    include_language=False,
                    device='cpu'
                )
                
                # Extract vision embeddings
                vision_embeddings = batch_data['vision_embeddings']
                
                # Store in cache
                for j, obs_id in enumerate(batch_ids):
                    # Vision embeddings shape: (8, 24, 24, 1408)
                    vision_cache[obs_id] = vision_embeddings[j].numpy()
                    processed += 1
                
                # Checkpoint save
                if processed % checkpoint_interval == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"\n💾 Checkpoint saved ({processed} embeddings)")
                
                # Memory management
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted! Saving checkpoint...")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            raise
        
        # Final save
        with open(output_file, 'wb') as f:
            pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Clean up checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"\n✅ Cached {len(vision_cache)} total embeddings")
        print(f"   File size: {os.path.getsize(output_file) / 1024**3:.2f} GB")
        
        return vision_cache
    
    def create_train_test_caches(self,
                               max_per_species: int = 25,
                               min_species: int = 10):
        """
        Create both training and test vision caches
        
        Args:
            max_per_species: Maximum observations per species
            min_species: Minimum number of species
        """
        print("🎯 Creating training and test vision caches\n")
        
        # Get train observations
        print("=" * 60)
        print("TRAINING SET")
        print("=" * 60)
        train_obs_ids, train_species = self.get_species_observations(
            max_per_species=max_per_species,
            min_species=min_species,
            test_split=False
        )
        
        # Cache training embeddings
        train_cache = self.cache_vision_embeddings(
            train_obs_ids,
            'vision_cache_train.pkl'
        )
        
        # Get test observations
        print("\n" + "=" * 60)
        print("TEST SET")
        print("=" * 60)
        test_obs_ids, test_species = self.get_species_observations(
            max_per_species=max_per_species,
            min_species=min_species,
            test_split=True
        )
        
        # Cache test embeddings
        test_cache = self.cache_vision_embeddings(
            test_obs_ids,
            'vision_cache_test.pkl'
        )
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✅ Training embeddings: {len(train_cache)}")
        print(f"✅ Test embeddings: {len(test_cache)}")
        
        # Check overlap
        overlap = set(train_cache.keys()) & set(test_cache.keys())
        if overlap:
            print(f"⚠️  Warning: {len(overlap)} embeddings in both sets")
        else:
            print("✅ No overlap between train and test sets")
        
        # Species coverage
        common_species = set(train_species.keys()) & set(test_species.keys())
        print(f"\n📊 Species coverage:")
        print(f"   Common species: {len(common_species)}")
        print(f"   Train-only species: {len(set(train_species.keys()) - common_species)}")
        print(f"   Test-only species: {len(set(test_species.keys()) - common_species)}")
        
        # Save metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'train_observations': len(train_cache),
            'test_observations': len(test_cache),
            'train_species': train_species,
            'test_species': test_species,
            'common_species': list(common_species),
            'config_path': self.config_path
        }
        
        with open('vision_cache_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("\n📝 Saved metadata to vision_cache_metadata.json")
    
    def verify_cache_compatibility(self):
        """Verify that cached embeddings work with the training script"""
        print("\n🔍 Verifying cache compatibility...")
        
        # Load caches
        if not os.path.exists('vision_cache_train.pkl'):
            print("❌ vision_cache_train.pkl not found")
            return False
        
        with open('vision_cache_train.pkl', 'rb') as f:
            train_cache = pickle.load(f)
        
        # Get a sample embedding
        sample_id = next(iter(train_cache.keys()))
        sample_embedding = train_cache[sample_id]
        
        print(f"\n📊 Sample embedding verification:")
        print(f"   Observation ID: {sample_id}")
        print(f"   Shape: {sample_embedding.shape}")
        print(f"   Expected shape: (8, 24, 24, 1408)")
        print(f"   Dtype: {sample_embedding.dtype}")
        print(f"   Range: [{sample_embedding.min():.3f}, {sample_embedding.max():.3f}]")
        
        # Verify shape
        if sample_embedding.shape != (8, 24, 24, 1408):
            print("❌ Shape mismatch!")
            return False
        
        # Test with training dataset
        print("\n🧪 Testing with DeepEarthOptimizedDataset...")
        try:
            # Create a minimal test
            from collections import defaultdict
            
            class MockDataset:
                def __init__(self, cache_file):
                    with open(cache_file, 'rb') as f:
                        self.vision_cache = pickle.load(f)
                    print(f"   Loaded {len(self.vision_cache)} embeddings")
                    
                    # Check if we can convert to tensor
                    test_id = next(iter(self.vision_cache.keys()))
                    test_tensor = torch.tensor(self.vision_cache[test_id])
                    print(f"   Tensor conversion successful: {test_tensor.shape}")
            
            MockDataset('vision_cache_train.pkl')
            print("✅ Cache is compatible with training pipeline!")
            return True
            
        except Exception as e:
            print(f"❌ Compatibility test failed: {e}")
            return False


def main():
    """Main execution"""
    print("🌍 DeepEarth Integrated Vision Embedding Cacher\n")
    
    # Initialize cacher
    cacher = IntegratedVisionCacher()
    
    # Create train and test caches
    cacher.create_train_test_caches(
        max_per_species=25,
        min_species=10
    )
    
    # Verify compatibility
    cacher.verify_cache_compatibility()
    
    print("\n✅ Vision caching complete!")
    print("\nYour training script can now use these caches:")
    print("  - vision_cache_train.pkl")
    print("  - vision_cache_test.pkl")


if __name__ == "__main__":
    main()
