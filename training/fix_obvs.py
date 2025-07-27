#!/usr/bin/env python3
"""
Fixed parallel vision cacher using ProcessPoolExecutor
No more os.chdir() race conditions!
"""

import pickle
import pandas as pd
from pathlib import Path
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from multiprocessing import get_context

# Global for dashboard path
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

# Worker initialization (runs once per process)
def worker_init():
    """Initialize data cache in each worker process"""
    global data_cache
    
    # Add dashboard to path
    sys.path.insert(0, str(DASHBOARD_DIR))
    
    # Change to dashboard directory ONCE per process
    os.chdir(DASHBOARD_DIR)
    
    # Import here to avoid pickling issues
    from data_cache import UnifiedDataCache
    
    # Create cache instance for this process
    data_cache = UnifiedDataCache("dataset_config.json")
    

def load_batch_worker(batch_info):
    """Load a single batch of embeddings (runs in worker process)"""
    batch_idx, gbif_obs_pairs = batch_info
    
    try:
        gbif_ids = [gbif_id for gbif_id, _ in gbif_obs_pairs]
        obs_ids = [obs_id for _, obs_id in gbif_obs_pairs]
        
        # Import here to avoid pickling issues
        from services.training_data import get_training_batch
        
        # Time the loading
        start_time = time.time()
        
        # Load batch - no chdir needed, already in correct directory
        batch_data = get_training_batch(
            data_cache,  # Uses process-local cache
            obs_ids,
            include_vision=True,
            include_language=False,
            device='cpu'
        )
        
        load_time = time.time() - start_time
        
        # Extract embeddings
        vision_embeddings = batch_data['vision_embeddings'].numpy()
        
        # Return results
        results = []
        for j, gbif_id in enumerate(gbif_ids):
            results.append((gbif_id, vision_embeddings[j]))
        
        return batch_idx, results, load_time, None
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return batch_idx, None, 0, error_msg


def cache_all_parallel():
    """Main caching function using process pool"""
    
    print("⚡ Fixed Parallel Vision Cacher (ProcessPoolExecutor)\n")
    
    # Step 1: Get required GBIF IDs (in main process)
    print("📋 Getting required GBIF IDs...")
    obs_path = DASHBOARD_DIR / "huggingface_dataset/hf_download/observations.parquet"
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
    print(f"✅ Need {len(required_gbif_ids)} embeddings")
    
    # Step 2: Get observation IDs (in main process)
    print("\n🔍 Getting observation IDs...")
    
    # Import and initialize temporarily
    sys.path.insert(0, str(DASHBOARD_DIR))
    os.chdir(DASHBOARD_DIR)
    
    from services.training_data import get_available_observation_ids
    from data_cache import UnifiedDataCache
    
    temp_cache = UnifiedDataCache("dataset_config.json")
    all_obs_ids = get_available_observation_ids(
        temp_cache,
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
    
    # Return to original directory
    os.chdir(Path(__file__).parent)
    
    # Map to observation IDs
    gbif_obs_pairs = []
    for gbif_id in required_gbif_ids:
        gbif_str = str(gbif_id)
        if gbif_str in gbif_to_obs:
            gbif_obs_pairs.append((gbif_id, gbif_to_obs[gbif_str][0]))
    
    print(f"✅ Mapped {len(gbif_obs_pairs)} GBIF IDs to observation IDs")
    
    # Step 3: Load existing cache
    vision_cache = {}
    if os.path.exists('vision_cache_train.pkl'):
        print("\n📂 Loading existing cache...")
        with open('vision_cache_train.pkl', 'rb') as f:
            vision_cache = pickle.load(f)
        print(f"   Found {len(vision_cache)} existing embeddings")
        
        # Filter out already cached
        gbif_obs_pairs = [(g, o) for g, o in gbif_obs_pairs if g not in vision_cache]
        print(f"   Need to cache {len(gbif_obs_pairs)} new embeddings")
    
    if not gbif_obs_pairs:
        print("\n✅ All embeddings already cached!")
        return
    
    # Step 4: Parallel loading with ProcessPoolExecutor
    print(f"\n🚀 Loading {len(gbif_obs_pairs)} embeddings with process pool...")
    
    # Create batches
    batch_size = 16  # Smaller batches for stability
    batches = []
    for i in range(0, len(gbif_obs_pairs), batch_size):
        batch = gbif_obs_pairs[i:i + batch_size]
        batches.append((i // batch_size, batch))
    
    print(f"   Created {len(batches)} batches of size {batch_size}")
    
    # Process with pool
    num_workers = 4
    completed = 0
    failed = 0
    total_time = 0
    
    # Use spawn context for cleaner process creation
    ctx = get_context('spawn')
    
    with ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=worker_init,
        mp_context=ctx
    ) as executor:
        
        # Submit all tasks
        future_to_batch = {
            executor.submit(load_batch_worker, batch): batch[0] 
            for batch in batches
        }
        
        # Process results as they complete
        with tqdm(total=len(batches), desc="Loading batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_idx, results, load_time, error = future.result(timeout=300)  # 5 min timeout
                    
                    if error:
                        print(f"\n❌ Batch {batch_idx} failed: {error}")
                        failed += len(batches[batch_idx][1])
                    else:
                        # Store results
                        for gbif_id, embedding in results:
                            vision_cache[gbif_id] = embedding
                        
                        completed += len(results)
                        total_time += load_time
                        
                        # Save checkpoint every 10 batches
                        if (batch_idx + 1) % 10 == 0:
                            checkpoint_file = f'vision_cache_checkpoint_{len(vision_cache)}.pkl'
                            with open(checkpoint_file, 'wb') as f:
                                pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                            tqdm.write(f"💾 Checkpoint saved: {checkpoint_file}")
                        
                        # Update progress bar
                        avg_time = total_time / (pbar.n + 1)
                        pbar.set_postfix({
                            'cached': completed,
                            'failed': failed,
                            'avg_time': f'{avg_time:.1f}s',
                            'est_remaining': f'{avg_time * (len(batches) - pbar.n - 1) / 60:.1f}m'
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n❌ Error processing batch {batch_idx}: {e}")
                    failed += len(batches[batch_idx][1])
                    pbar.update(1)
    
    # Step 5: Save final cache
    print(f"\n💾 Saving final cache with {len(vision_cache)} embeddings...")
    
    output_file = 'vision_cache_train_fixed.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(vision_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_gb = os.path.getsize(output_file) / 1024**3
    print(f"✅ Saved to {output_file}")
    print(f"   Size: {file_size_gb:.2f} GB")
    print(f"   Successfully cached: {completed}")
    print(f"   Failed: {failed}")
    
    # Replace original if successful
    if len(vision_cache) > 256:
        if os.path.exists('vision_cache_train.pkl'):
            os.rename('vision_cache_train.pkl', 'vision_cache_train_old.pkl')
        os.rename(output_file, 'vision_cache_train.pkl')
        print("\n✅ Replaced vision_cache_train.pkl with fixed version")
    
    # Final stats
    if completed > 0:
        print(f"\n📊 Performance Statistics:")
        print(f"   Total embeddings cached: {len(vision_cache)}")
        print(f"   Total time: {total_time / 60:.1f} minutes")
        print(f"   Average time per batch: {total_time / len(batches):.1f} seconds")
        print(f"   Average time per embedding: {total_time / completed:.1f} seconds")
    
    # Check specific IDs
    check_ids = [3044830298, 3391675877, 3858956662, 4903847224, 4161982655]
    print("\n🔍 Checking previously missing IDs:")
    for gbif_id in check_ids:
        if gbif_id in vision_cache:
            print(f"   ✅ {gbif_id} now cached")
        else:
            print(f"   ❌ {gbif_id} still missing")
    
    print("\n✨ Caching complete!")


if __name__ == "__main__":
    # Increase file descriptor limit
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
        print(f"📝 Increased file descriptor limit to 4096")
    except:
        pass
    
    cache_all_parallel()
