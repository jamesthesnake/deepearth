#!/usr/bin/env python3
"""Analyze the trained cross-modal model"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from pathlib import Path

# Import model
from deepearth_crossmodal_training_optimized import CrossModalMLP, DeepEarthOptimizedDataset

def analyze_embeddings(model_path='deepearth_crossmodal_optimized.pth'):
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossModalMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load test data
    dataset = DeepEarthOptimizedDataset(
        max_per_species=10,
        test_split=True,
        cache_file='vision_cache_test.pkl'
    )
    
    # Extract embeddings
    vision_embeddings = []
    language_embeddings = []
    reconstructed_embeddings = []
    species_names = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            vision = sample['vision_embedding'].unsqueeze(0).to(device)
            language = sample['language_embedding'].unsqueeze(0).to(device)
            
            outputs = model(vision, language, mask_prob=0)
            
            vision_embeddings.append(outputs['vision_universal'].cpu().numpy())
            language_embeddings.append(outputs['language_universal'].cpu().numpy())
            reconstructed_embeddings.append(outputs['language_reconstructed'].cpu().numpy())
            species_names.append(sample['species'])
    
    vision_embeddings = np.vstack(vision_embeddings)
    language_embeddings = np.vstack(language_embeddings)
    
    # Calculate alignment
    print("\nCross-modal alignment:")
    alignments = []
    for i in range(len(vision_embeddings)):
        cos_sim = np.dot(vision_embeddings[i], language_embeddings[i]) / (
            np.linalg.norm(vision_embeddings[i]) * np.linalg.norm(language_embeddings[i])
        )
        alignments.append(cos_sim)
    
    print(f"  Mean alignment: {np.mean(alignments):.3f}")
    print(f"  Std alignment: {np.std(alignments):.3f}")
    
    # UMAP visualization
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Vision embeddings
    reducer = umap.UMAP(n_components=2, random_state=42)
    vision_2d = reducer.fit_transform(vision_embeddings)
    
    ax = axes[0]
    unique_species = list(set(species_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_species)))
    
    for i, species in enumerate(unique_species):
        mask = [s == species for s in species_names]
        ax.scatter(vision_2d[mask, 0], vision_2d[mask, 1], 
                  color=colors[i], label=species.split()[0], alpha=0.7)
    ax.set_title('Vision Embeddings (UMAP)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Language embeddings
    language_2d = reducer.fit_transform(language_embeddings)
    ax = axes[1]
    for i, species in enumerate(unique_species):
        mask = [s == species for s in species_names]
        ax.scatter(language_2d[mask, 0], language_2d[mask, 1], 
                  color=colors[i], label=species.split()[0], alpha=0.7)
    ax.set_title('Language Embeddings (UMAP)')
    
    # 3. Alignment plot
    ax = axes[2]
    ax.hist(alignments, bins=30, alpha=0.7)
    ax.axvline(np.mean(alignments), color='red', linestyle='--', label=f'Mean: {np.mean(alignments):.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Cross-Modal Alignment')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('embedding_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved analysis to embedding_analysis.png")
    
    # Species similarity matrix
    print("\nComputing species similarity matrix...")
    species_embeddings = {}
    for species in unique_species:
        mask = [s == species for s in species_names]
        species_embeddings[species] = vision_embeddings[mask].mean(axis=0)
    
    n_species = len(unique_species)
    similarity_matrix = np.zeros((n_species, n_species))
    
    for i, sp1 in enumerate(unique_species):
        for j, sp2 in enumerate(unique_species):
            sim = np.dot(species_embeddings[sp1], species_embeddings[sp2]) / (
                np.linalg.norm(species_embeddings[sp1]) * np.linalg.norm(species_embeddings[sp2])
            )
            similarity_matrix[i, j] = sim
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, 
                xticklabels=[s.split()[0] for s in unique_species],
                yticklabels=[s.split()[0] for s in unique_species],
                cmap='coolwarm', center=0.5, 
                annot=True, fmt='.2f')
    plt.title('Species Similarity Matrix (Vision Embeddings)')
    plt.tight_layout()
    plt.savefig('species_similarity.png', dpi=150)
    print("Saved species similarity to species_similarity.png")

if __name__ == "__main__":
    analyze_embeddings()
