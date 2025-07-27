#!/usr/bin/env python3
"""
Analyze and clean DeepEarth dataset splits properly
Uses actual train/test splits and creates balanced, verifiable datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import argparse


def load_split_info(data_dir):
    """Load actual split information from existing files or dataset"""
    data_dir = Path(data_dir)
    
    # Try to find existing split file
    split_files = [
        data_dir / 'central_florida_split.json',
        data_dir / 'split_info.json',
        data_dir / 'dataset_info.json'
    ]
    
    for split_file in split_files:
        if split_file.exists():
            print(f"Loading split info from: {split_file}")
            with open(split_file, 'r') as f:
                return json.load(f)
    
    print("No split file found, inferring from data...")
    return None


def analyze_dataset_splits(data_dir, min_train=5, min_test=3, output_dir=None):
    """
    Analyze train/test splits properly and create cleaned datasets
    
    Args:
        data_dir: Path to dataset directory
        min_train: Minimum samples required in train set
        min_test: Minimum samples required in test set
        output_dir: Where to save outputs (defaults to data_dir/analysis)
    """
    
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir / 'analysis'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load observations
    print("Loading observations...")
    obs_df = pd.read_parquet(data_dir / 'observations.parquet')
    
    # Filter for vision data
    obs_with_vision = obs_df[obs_df['has_vision'] == True].copy()
    print(f"Total observations with vision: {len(obs_with_vision)}")
    
    # Efficient species counting using groupby
    species_counts = obs_with_vision.groupby('taxon_name').size()
    print(f"\nTotal unique species: {len(species_counts)}")
    print(f"Species count range: {species_counts.min()} - {species_counts.max()}")
    
    # Load or infer split information
    split_info = load_split_info(data_dir)
    
    # Analyze actual train/test distribution
    train_obs = []
    test_obs = []
    species_split_counts = defaultdict(lambda: {'train': 0, 'test': 0, 'total': 0})
    
    # If we have split column, use it
    if 'split' in obs_with_vision.columns:
        print("\nUsing 'split' column from dataset")
        train_mask = obs_with_vision['split'] == 'train'
        test_mask = obs_with_vision['split'] == 'test'
        
        train_obs = obs_with_vision[train_mask]
        test_obs = obs_with_vision[test_mask]
        
        # Count per species
        train_species_counts = train_obs.groupby('taxon_name').size()
        test_species_counts = test_obs.groupby('taxon_name').size()
        
    else:
        # Infer split based on observation order and counts
        print("\nInferring split from observation order...")
        
        # Group by species and assign train/test based on position
        for species, group in obs_with_vision.groupby('taxon_name'):
            n_samples = len(group)
            species_split_counts[species]['total'] = n_samples
            
            # Common splitting strategies
            if n_samples >= 30:
                # Can have both train and test
                n_train = 20
                n_test = min(10, n_samples - 20)
            elif n_samples >= 10:
                # Limited test samples
                n_train = int(n_samples * 0.7)
                n_test = n_samples - n_train
            else:
                # Too few samples, all go to train
                n_train = n_samples
                n_test = 0
            
            # Assign splits
            group_sorted = group.sort_values('gbif_id')  # Consistent ordering
            train_idx = group_sorted.index[:n_train]
            test_idx = group_sorted.index[n_train:n_train + n_test]
            
            train_obs.append(obs_with_vision.loc[train_idx])
            if n_test > 0:
                test_obs.append(obs_with_vision.loc[test_idx])
            
            species_split_counts[species]['train'] = n_train
            species_split_counts[species]['test'] = n_test
        
        # Combine
        train_obs = pd.concat(train_obs) if train_obs else pd.DataFrame()
        test_obs = pd.concat(test_obs) if test_obs else pd.DataFrame()
        
        train_species_counts = train_obs.groupby('taxon_name').size()
        test_species_counts = test_obs.groupby('taxon_name').size()
    
    # Find species meeting minimum requirements
    species_with_enough_train = set(train_species_counts[train_species_counts >= min_train].index)
    species_with_enough_test = set(test_species_counts[test_species_counts >= min_test].index)
    species_to_keep = species_with_enough_train & species_with_enough_test
    
    print(f"\nSpecies analysis:")
    print(f"  With >= {min_train} train samples: {len(species_with_enough_train)}")
    print(f"  With >= {min_test} test samples: {len(species_with_enough_test)}")
    print(f"  Meeting both requirements: {len(species_to_keep)}")
    
    # Show what we're removing
    removed_species = set(species_counts.index) - species_to_keep
    print(f"\nRemoving {len(removed_species)} species:")
    
    # Analyze why species are removed
    removed_low_train = removed_species - species_with_enough_train
    removed_low_test = removed_species - species_with_enough_test
    removed_both = removed_low_train & removed_low_test
    
    print(f"  Too few train samples only: {len(removed_low_train - removed_both)}")
    print(f"  Too few test samples only: {len(removed_low_test - removed_both)}")
    print(f"  Too few in both: {len(removed_both)}")
    
    # Create cleaned dataset
    cleaned_obs = obs_with_vision[obs_with_vision['taxon_name'].isin(species_to_keep)]
    
    # Recompute train/test assignments for cleaned data
    cleaned_train = train_obs[train_obs['taxon_name'].isin(species_to_keep)]
    cleaned_test = test_obs[test_obs['taxon_name'].isin(species_to_keep)]
    
    # Verify balance
    cleaned_train_counts = cleaned_train.groupby('taxon_name').size()
    cleaned_test_counts = cleaned_test.groupby('taxon_name').size()
    
    # Calculate train:test ratios
    ratios = []
    manifest = {}
    
    for species in species_to_keep:
        train_n = cleaned_train_counts.get(species, 0)
        test_n = cleaned_test_counts.get(species, 0)
        ratio = train_n / test_n if test_n > 0 else np.inf
        ratios.append(ratio)
        
        manifest[species] = {
            'train': int(train_n),
            'test': int(test_n),
            'total': int(train_n + test_n),
            'ratio': float(ratio)
        }
    
    ratios = np.array(ratios)
    print(f"\nTrain:Test ratio statistics:")
    print(f"  Mean: {np.mean(ratios[ratios != np.inf]):.2f}")
    print(f"  Median: {np.median(ratios[ratios != np.inf]):.2f}")
    print(f"  Min: {np.min(ratios[ratios != np.inf]):.2f}")
    print(f"  Max: {np.max(ratios[ratios != np.inf]):.2f}")
    
    # Save outputs
    print(f"\nSaving cleaned dataset...")
    
    # Save cleaned observations with split column
    cleaned_obs_with_split = cleaned_obs.copy()
    cleaned_obs_with_split['split'] = 'unknown'
    cleaned_obs_with_split.loc[cleaned_train.index, 'split'] = 'train'
    cleaned_obs_with_split.loc[cleaned_test.index, 'split'] = 'test'
    
    output_path = output_dir / 'observations_cleaned.parquet'
    cleaned_obs_with_split.to_parquet(output_path)
    print(f"  Saved observations to: {output_path}")
    
    # Save manifest
    manifest_path = output_dir / 'species_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to: {manifest_path}")
    
    # Save summary
    summary = {
        'original_species': int(len(species_counts)),
        'cleaned_species': int(len(species_to_keep)),
        'removed_species': int(len(removed_species)),
        'original_observations': int(len(obs_with_vision)),
        'cleaned_observations': int(len(cleaned_obs)),
        'train_observations': int(len(cleaned_train)),
        'test_observations': int(len(cleaned_test)),
        'min_train_threshold': min_train,
        'min_test_threshold': min_test,
        'mean_train_test_ratio': float(np.mean(ratios[ratios != np.inf])),
        'removed_species_list': sorted(list(removed_species))
    }
    
    summary_path = output_dir / 'cleaning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to: {summary_path}")
    
    # Plot distributions
    plot_distributions(
        species_counts,
        cleaned_train_counts,
        cleaned_test_counts,
        removed_species,
        output_dir
    )
    
    return summary


def plot_distributions(original_counts, train_counts, test_counts, removed_species, output_dir):
    """Create comprehensive distribution plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original distribution
    ax = axes[0, 0]
    counts = original_counts.values
    kept_mask = ~original_counts.index.isin(removed_species)
    
    ax.hist(counts[kept_mask], bins=30, alpha=0.7, color='green', label='Kept species', edgecolor='black')
    ax.hist(counts[~kept_mask], bins=30, alpha=0.7, color='red', label='Removed species', edgecolor='black')
    ax.set_xlabel('Total Samples per Species')
    ax.set_ylabel('Number of Species')
    ax.set_title('Original Species Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # Train distribution
    ax = axes[0, 1]
    ax.hist(train_counts.values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Train Samples per Species')
    ax.set_ylabel('Number of Species')
    ax.set_title('Cleaned Train Distribution')
    ax.set_yscale('log')
    
    # Test distribution
    ax = axes[1, 0]
    ax.hist(test_counts.values, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Test Samples per Species')
    ax.set_ylabel('Number of Species')
    ax.set_title('Cleaned Test Distribution')
    ax.set_yscale('log')
    
    # Train:Test ratios
    ax = axes[1, 1]
    ratios = []
    for species in train_counts.index:
        if species in test_counts:
            ratio = train_counts[species] / test_counts[species]
            ratios.append(min(ratio, 10))  # Cap at 10 for visualization
    
    ax.hist(ratios, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=2, color='red', linestyle='--', label='2:1 ratio')
    ax.axvline(x=3, color='orange', linestyle='--', label='3:1 ratio')
    ax.set_xlabel('Train:Test Ratio (capped at 10)')
    ax.set_ylabel('Number of Species')
    ax.set_title('Train:Test Ratio Distribution')
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / 'dataset_distribution_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and clean dataset splits',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data-dir/analysis)')
    parser.add_argument('--min-train', type=int, default=5,
                       help='Minimum train samples per species (default: 5)')
    parser.add_argument('--min-test', type=int, default=3,
                       help='Minimum test samples per species (default: 3)')
    
    args = parser.parse_args()
    
    # Analyze and clean dataset
    summary = analyze_dataset_splits(
        args.data_dir,
        min_train=args.min_train,
        min_test=args.min_test,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("Dataset Cleaning Complete!")
    print("="*60)
    print(f"Original species: {summary['original_species']}")
    print(f"Cleaned species: {summary['cleaned_species']}")
    print(f"Removed species: {summary['removed_species']}")
    print(f"\nTo use the cleaned dataset, your training script will")
    print(f"automatically detect and use: {Path(args.output_dir or args.data_dir) / 'analysis/observations_cleaned.parquet'}")


if __name__ == "__main__":
    main()
