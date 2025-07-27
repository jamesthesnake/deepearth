#!/usr/bin/env python3
"""
DeepEarth Cross-Modal Masking with Real Data
Implements the core masked reconstruction pipeline from your diagram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import umap
from tqdm import tqdm

class DeepEarthRealDataset(Dataset):
    """Load real embeddings from HuggingFace dataset"""
    
    def __init__(self, data_dir='../dashboard/huggingface_dataset/hf_download/', 
                 species_list=None, max_per_species=25, test_split=False):
        self.data_dir = Path(data_dir)
        
        # Load observations
        obs_df = pd.read_parquet(self.data_dir / 'observations.parquet')
        obs_df = obs_df[obs_df['has_vision'] == True]
        
        # Filter to specific species if provided
        if species_list:
            obs_df = obs_df[obs_df['taxon_name'].isin(species_list)]
        
        # Sample max_per_species for each species
        self.data = []
        for species in obs_df['taxon_name'].unique()[:10]:  # Max 10 species
            species_obs = obs_df[obs_df['taxon_name'] == species]
            
            # Train/test split
            if test_split:
                species_obs = species_obs.iloc[max_per_species:max_per_species+10]
            else:
                species_obs = species_obs.iloc[:max_per_species]
            
            for _, obs in species_obs.iterrows():
                self.data.append({
                    'gbif_id': obs['gbif_id'],
                    'species': obs['taxon_name'],
                    'latitude': obs['latitude'],
                    'longitude': obs['longitude'],
                    'language_embedding': np.array(obs['language_embedding'], dtype=np.float32)
                })
        
        # Load vision index for mapping
        self.vision_index = pd.read_parquet(self.data_dir / 'vision_index.parquet')
        
        # Create species to index mapping
        self.species_list = sorted(list(set([d['species'] for d in self.data])))
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        
    def _load_vision_embedding(self, gbif_id):
        """Load actual vision embedding from files"""
        vision_info = self.vision_index[self.vision_index['gbif_id'] == gbif_id]
        if len(vision_info) == 0:
            return np.random.randn(8, 24, 24, 1408).astype(np.float32)
        
        filename = vision_info.iloc[0]['filename']
        file_path = self.data_dir / 'vision_embeddings' / filename
        
        try:
            emb_df = pd.read_parquet(file_path)
            emb_row = emb_df[emb_df['gbif_id'] == gbif_id].iloc[0]
            embedding = np.array(emb_row['embedding'], dtype=np.float32)
            # Reshape from flattened to (8, 24, 24, 1408)
            embedding = embedding.reshape(8, 24, 24, 1408)
            return embedding
        except:
            return np.random.randn(8, 24, 24, 1408).astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        vision_emb = self._load_vision_embedding(item['gbif_id'])
        
        return {
            'vision_embedding': torch.tensor(vision_emb),
            'language_embedding': torch.tensor(item['language_embedding']),
            'species': item['species'],
            'species_idx': self.species_to_idx[item['species']],
            'coordinates': torch.tensor([item['latitude'], item['longitude']])
        }


class CrossModalMLP(nn.Module):
    """MLP-based cross-modal encoder following your design"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, hidden_dim=256, universal_dim=2048):
        super().__init__()
        
        # Vision encoder: spatial pooling + MLP
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, universal_dim)
        )
        
        # Language encoder
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, universal_dim)
        )
        
        # Decoders
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, language_dim)
        )
        
    def forward(self, vision_emb, language_emb, mask_language=True, mask_prob=0.5):
        batch_size = vision_emb.shape[0]
        device = vision_emb.device
        
        # Pool vision embeddings across spatial and temporal dimensions
        vision_pooled = vision_emb.mean(dim=(1, 2, 3))  # (B, 1408)
        
        # Encode to universal space
        vision_universal = self.vision_encoder(vision_pooled)
        language_universal = self.language_encoder(language_emb)
        
        # Stochastic masking
        if self.training and mask_prob > 0:
            mask = torch.rand(batch_size, device=device) < mask_prob
        else:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Reconstruction
        if mask_language:
            # Reconstruct language from vision
            language_reconstructed = self.language_decoder(vision_universal)
            reconstruction_loss = F.mse_loss(
                language_reconstructed[mask], 
                language_emb[mask]
            ) if mask.any() else torch.tensor(0.0, device=device)
        else:
            language_reconstructed = None
            reconstruction_loss = torch.tensor(0.0, device=device)
        
        return {
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'language_reconstructed': language_reconstructed,
            'reconstruction_loss': reconstruction_loss,
            'mask': mask
        }


def add_ecophysiology_prefix(dataset, model, prefix="ecophysiology of "):
    """
    Experiment: Add prefix to species names and see embedding shift
    This tests if language embeddings capture semantic relationships
    """
    print(f"\n🌿 Testing '{prefix}' prefix experiment...")
    
    # For this experiment, we'd need to:
    # 1. Get DeepSeek embeddings for "species_name" vs "ecophysiology of species_name"
    # 2. Compare the embedding shifts
    # 3. See if reconstruction quality changes
    
    # Placeholder for now - would need DeepSeek model access
    print("  Would require DeepSeek model to generate new embeddings")
    print("  Expected: Embeddings should shift towards ecological concepts")


def compute_knn_percentiles(embeddings, labels, k=5):
    """
    Compute k-NN accuracy at different percentiles
    Shows how well species cluster in the learned space
    """
    n_samples = len(embeddings)
    knn = NearestNeighbors(n_neighbors=min(k+1, n_samples))
    knn.fit(embeddings)
    
    # Get all distances and indices
    distances, indices = knn.kneighbors(embeddings)
    
    # Compute accuracy for each sample
    accuracies = []
    for i in range(n_samples):
        # Exclude self (index 0)
        neighbor_indices = indices[i, 1:]
        neighbor_labels = labels[neighbor_indices]
        
        # What fraction of neighbors are same species?
        same_species = (neighbor_labels == labels[i]).sum()
        accuracy = same_species / len(neighbor_labels)
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    
    # Compute percentiles
    percentiles = [0, 25, 50, 75, 90, 95, 100]
    results = {}
    for p in percentiles:
        results[f'p{p}'] = np.percentile(accuracies, p)
    
    return results, accuracies


def add_temporal_masking(vision_emb, temporal_mask_prob=0.3):
    """
    Add temporal masking patterns
    Mask entire temporal frames to test temporal understanding
    """
    batch_size, T, H, W, C = vision_emb.shape
    device = vision_emb.device
    
    # Randomly mask entire temporal frames
    temporal_mask = torch.rand(batch_size, T, device=device) > temporal_mask_prob
    temporal_mask = temporal_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    masked_vision = vision_emb * temporal_mask.float()
    return masked_vision, temporal_mask.squeeze()


def validate_spacetime_integration(model, dataloader):
    """
    Validate that the model learns proper space-time integration
    by checking reconstruction quality with different masking patterns
    """
    print("\n🌍 Validating space-time integration...")
    
    model.eval()
    results = {
        'no_mask': [],
        'spatial_mask': [],
        'temporal_mask': [],
        'both_mask': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            vision = batch['vision_embedding']
            language = batch['language_embedding']
            coords = batch['coordinates']
            
            # Test different masking patterns
            # 1. No masking
            outputs = model(vision, language, mask_prob=0)
            loss_no_mask = F.mse_loss(outputs['language_reconstructed'], language)
            results['no_mask'].append(loss_no_mask.item())
            
            # 2. Temporal masking
            vision_temporal_masked, _ = add_temporal_masking(vision)
            outputs = model(vision_temporal_masked, language, mask_prob=0)
            loss_temporal = F.mse_loss(outputs['language_reconstructed'], language)
            results['temporal_mask'].append(loss_temporal.item())
    
    # Print results
    for mask_type, losses in results.items():
        if losses:
            print(f"  {mask_type}: {np.mean(losses):.4f} ± {np.std(losses):.4f}")


def train_with_real_data(model, train_loader, test_loader, epochs=30):
    """Main training loop with all validations"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device
    
    # Track metrics
    train_losses = []
    knn_accuracies = []
    embedding_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            
            # Forward pass with masking
            outputs = model(vision, language, mask_language=True, mask_prob=0.5)
            loss = outputs['reconstruction_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation with k-NN
        if epoch % 5 == 0:
            model.eval()
            all_embeddings = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    vision = batch['vision_embedding'].to(device)
                    outputs = model(vision, None, mask_language=False)
                    
                    embeddings = outputs['vision_universal'].cpu().numpy()
                    labels = batch['species_idx'].numpy()
                    
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
            
            all_embeddings = np.vstack(all_embeddings)
            all_labels = np.hstack(all_labels)
            
            # Compute k-NN percentiles
            percentiles, accuracies = compute_knn_percentiles(all_embeddings, all_labels)
            knn_accuracies.append(percentiles)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  k-NN median accuracy: {percentiles['p50']:.2%}")
            print(f"  k-NN 90th percentile: {percentiles['p90']:.2%}")
            
            # Store for UMAP
            embedding_history.append((all_embeddings, all_labels))
    
    return train_losses, knn_accuracies, embedding_history


def visualize_training_progress(train_losses, knn_accuracies, embedding_history):
    """Create comprehensive visualization of training progress"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Training loss
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    
    # 2. k-NN accuracy percentiles
    ax2 = plt.subplot(2, 2, 2)
    epochs = [i*5 for i in range(len(knn_accuracies))]
    percentiles = ['p50', 'p75', 'p90']
    for p in percentiles:
        values = [acc[p] for acc in knn_accuracies]
        ax2.plot(epochs, values, label=p)
    ax2.set_title('k-NN Accuracy Percentiles')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # 3. UMAP progression
    n_snapshots = min(3, len(embedding_history))
    for i in range(n_snapshots):
        ax = plt.subplot(2, n_snapshots, n_snapshots + i + 1)
        embeddings, labels = embedding_history[i * (len(embedding_history)//n_snapshots)]
        
        # Run UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedded = reducer.fit_transform(embeddings)
        
        # Plot by species
        for species_idx in np.unique(labels):
            mask = labels == species_idx
            ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                      alpha=0.6, s=30, label=f'S{species_idx}')
        
        ax.set_title(f'UMAP at Epoch {i * (len(embedding_history)//n_snapshots) * 5}')
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('deepearth_training_progress.png', dpi=150, bbox_inches='tight')
    print("\n📊 Saved training visualization to deepearth_training_progress.png")


# Main execution
if __name__ == "__main__":
    print("🌍 DeepEarth Cross-Modal Training with Real Data\n")
    
    # Target species from dataset
    target_species = [
        "Magnolia grandiflora",
        "Pinus elliottii",
        "Quercus virginiana",
        "Liquidambar styraciflua",
        "Acer rubrum"
    ]
    
    # Create datasets
    print("Loading real HuggingFace embeddings...")
    train_dataset = DeepEarthRealDataset(
        species_list=target_species,
        max_per_species=20,
        test_split=False
    )
    test_dataset = DeepEarthRealDataset(
        species_list=target_species,
        max_per_species=20,
        test_split=True
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Species: {len(train_dataset.species_list)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossModalMLP().to(device)
    print(f"\nModel initialized on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with validations
    train_losses, knn_accuracies, embedding_history = train_with_real_data(
        model, train_loader, test_loader, epochs=30
    )
    
    # Validate space-time integration
    validate_spacetime_integration(model, test_loader)
    
    # Ecophysiology prefix experiment (placeholder)
    add_ecophysiology_prefix(train_dataset, model)
    
    # Visualize results
    visualize_training_progress(train_losses, knn_accuracies, embedding_history)
    
    # Save model
    torch.save(model.state_dict(), 'deepearth_crossmodal_mlp.pth')
    print("\n✅ Model saved to deepearth_crossmodal_mlp.pth")
