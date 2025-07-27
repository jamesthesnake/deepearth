#!/usr/bin/env python3
"""
Download the complete DeepEarth dataset including vision embeddings
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import os
from tqdm import tqdm

print("🌍 Downloading DeepEarth Central Florida Native Plants dataset...")

# Create directory structure
base_dir = Path("dashboard/huggingface_dataset/hf_download")
base_dir.mkdir(parents=True, exist_ok=True)

# Download the main dataset
print("\n1. Downloading observations...")
dataset = load_dataset("deepearth/central-florida-native-plants")

# Save observations
train_df = dataset['train'].to_pandas()
train_df.to_parquet(base_dir / "observations.parquet")
print(f"✓ Saved {len(train_df)} observations")

# List all files in the repository to find vision embeddings
print("\n2. Checking for vision embeddings in repository...")
repo_files = list_repo_files("deepearth/central-florida-native-plants", repo_type="dataset")

# Find vision-related files
vision_files = [f for f in repo_files if 'vision' in f.lower() or 'embedding' in f.lower()]
print(f"Found {len(vision_files)} vision-related files")

# Download vision index if it exists
vision_index_files = [f for f in repo_files if 'vision_index' in f]
if vision_index_files:
    print("\n3. Downloading vision index...")
    for f in vision_index_files:
        downloaded_file = hf_hub_download(
            repo_id="deepearth/central-florida-native-plants",
            filename=f,
            repo_type="dataset",
            cache_dir=base_dir.parent
        )
        # Copy to correct location
        target = base_dir / Path(f).name
        os.system(f"cp '{downloaded_file}' '{target}'")
        print(f"✓ Downloaded {f}")

# Download vision embedding files
vision_emb_dir = base_dir / "vision_embeddings"
vision_emb_dir.mkdir(exist_ok=True)

embedding_files = [f for f in repo_files if 'embeddings_' in f and f.endswith('.parquet')]
if embedding_files:
    print(f"\n4. Downloading {len(embedding_files)} vision embedding files...")
    for f in tqdm(embedding_files[:5]):  # Download first 5 as test
        downloaded_file = hf_hub_download(
            repo_id="deepearth/central-florida-native-plants",
            filename=f,
            repo_type="dataset",
            cache_dir=base_dir.parent
        )
        # Copy to correct location
        target = vision_emb_dir / Path(f).name
        os.system(f"cp '{downloaded_file}' '{target}'")
    print(f"✓ Downloaded {len(embedding_files[:5])} embedding files (sample)")
    print(f"   Note: Full dataset has {len(embedding_files)} embedding files")
else:
    print("⚠️  No vision embedding files found in repository")

# Print dataset info
print("\n📊 Dataset Statistics:")
print(f"Total observations: {len(train_df)}")
print(f"Observations with vision: {train_df['has_vision'].sum()}")
print(f"Unique species: {train_df['taxon_name'].nunique()}")
print(f"\nData saved to: {base_dir.absolute()}")

# Save a sample for testing
sample_df = train_df[train_df['has_vision']].head(100)
sample_df.to_parquet(base_dir / "observations_sample.parquet")
print(f"\n✓ Also saved a sample of 100 observations with vision for testing")
