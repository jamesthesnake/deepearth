#!/usr/bin/env python3
"""
DeepEarth Simple MLP U-Net Training - Dashboard Integrated Version
Uses the same data loading system as train_classifier.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
import random
import pandas as pd
import csv

# Add dashboard to path for data access
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepEarthMLPDataset(Dataset):
    """Dataset that loads data through the dashboard cache system"""
    
    def __init__(self, observation_ids, cache, mode='both', device='cpu', species_mapping=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.mode = mode
        self.device = device
        self.species_mapping = species_mapping
        
        # Load all data at initialization
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and prepare all data at initialization."""
        logger.info(f"Loading dataset with {len(self.observation_ids)} observations...")
        
        # Load data in batches
        batch_size = 64
        all_species = []
        all_language_embs = []
        all_vision_embs = []
        all_obs_ids = []
        
        for i in range(0, len(self.observation_ids), batch_size):
            batch_ids = self.observation_ids[i:i + batch_size]
            
            try:
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=(self.mode in ['vision', 'both']),
                    include_language=(self.mode in ['language', 'both']),
                    device='cpu'  # Load to CPU first
                )
                
                # The batch data has these keys based on train_classifier.py
                if 'species' in batch_data:
                    all_species.extend(batch_data['species'])
                if 'observation_ids' in batch_data:
                    all_obs_ids.extend(batch_data['observation_ids'])
                
                if self.mode in ['language', 'both'] and 'language_embeddings' in batch_data:
                    all_language_embs.append(batch_data['language_embeddings'])
                if self.mode in ['vision', 'both'] and 'vision_embeddings' in batch_data:
                    all_vision_embs.append(batch_data['vision_embeddings'])
                    
            except Exception as e:
                logger.warning(f"Error loading batch {i}: {e}")
                continue
        
        # Create species label mapping
        if self.species_mapping is not None:
            self.species_to_idx = self.species_mapping
        else:
            unique_species = sorted(list(set(all_species)))
            self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.num_classes = len(self.species_to_idx)
        
        # Filter data to only include mapped species
        filtered_indices = [i for i, species in enumerate(all_species) if species in self.species_to_idx]
        
        # Convert to tensors
        self.species_labels = torch.tensor(
            [self.species_to_idx[all_species[i]] for i in filtered_indices], 
            dtype=torch.long, device=self.device
        )
        self.observation_ids = [all_obs_ids[i] for i in filtered_indices]
        
        if self.mode in ['language', 'both'] and all_language_embs:
            all_language = torch.cat(all_language_embs, dim=0)
            self.language_embeddings = all_language[filtered_indices].to(self.device)
            
        if self.mode in ['vision', 'both'] and all_vision_embs:
            all_vision = torch.cat(all_vision_embs, dim=0)
            self.vision_embeddings = all_vision[filtered_indices].to(self.device)
        
        logger.info(f"Dataset loaded: {len(self.species_labels)} observations, {self.num_classes} species")
    
    def __len__(self):
        return len(self.species_labels)
    
    def __getitem__(self, idx):
        """Get single sample from dataset."""
        sample = {
            'species_idx': self.species_labels[idx],
            'species': self.idx_to_species[self.species_labels[idx].item()],
            'gbif_id': self.observation_ids[idx]
        }
        
        if self.mode in ['language', 'both']:
            sample['language_embedding'] = self.language_embeddings[idx]
            
        if self.mode in ['vision', 'both']:
            sample['vision_embedding'] = self.vision_embeddings[idx]
            
        return sample


class SimpleMLPUNet(nn.Module):
    """Simple MLP U-Net for cross-modal learning"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        
        # Encoder side
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, universal_dim),
            nn.ReLU(),
            nn.LayerNorm(universal_dim)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, universal_dim),
            nn.ReLU(),
            nn.LayerNorm(universal_dim)
        )
        
        # Universal bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(universal_dim, universal_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(universal_dim, universal_dim)
        )
        
        # Decoder side
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, 768),
            nn.ReLU(),
            nn.Linear(768, vision_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, 768),
            nn.ReLU(),
            nn.Linear(768, language_dim)
        )
    
    def encode_vision(self, vision_emb):
        """Encode vision embeddings to universal space"""
        # Pool over spatial and temporal dimensions if needed
        if vision_emb.ndim == 5:
            # (B, T, H, W, C) -> (B, C)
            vision_pooled = vision_emb.mean(dim=(1, 2, 3))
        else:
            vision_pooled = vision_emb
        
        return self.vision_encoder(vision_pooled)
    
    def encode_language(self, language_emb):
        """Encode language embeddings to universal space"""
        return self.language_encoder(language_emb)
    
    def forward(self, vision_emb, language_emb, mask_language=None, mode='both'):
        """Forward pass with optional language masking"""
        # Apply masking to language embeddings if provided
        if mask_language is not None and mode in ['language', 'both']:
            language_masked = language_emb.masked_fill(mask_language, 0.0)
        else:
            language_masked = language_emb
        
        # Encode to universal space
        vision_universal = self.encode_vision(vision_emb)
        language_universal = self.encode_language(language_masked)
        
        # Apply bottleneck with residual
        vision_universal = vision_universal + self.bottleneck(vision_universal)
        language_universal = language_universal + self.bottleneck(language_universal)
        
        # Decode back to original spaces
        vision_recon = self.vision_decoder(vision_universal)
        language_recon = self.language_decoder(language_universal)
        
        # Pool vision for reconstruction loss if needed
        if vision_emb.ndim == 5:
            vision_original_pooled = vision_emb.mean(dim=(1, 2, 3))
        else:
            vision_original_pooled = vision_emb
        
        return {
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_original': vision_original_pooled,
            'language_original': language_emb
        }


def compute_losses(outputs, compute_alignment=True):
    """Compute reconstruction and alignment losses"""
    losses = {}
    
    # L2/MSE reconstruction losses
    losses['recon_vision'] = F.mse_loss(outputs['vision_recon'], outputs['vision_original'])
    losses['recon_language'] = F.mse_loss(outputs['language_recon'], outputs['language_original'])
    losses['recon_total'] = losses['recon_vision'] + losses['recon_language']
    
    if compute_alignment:
        # Cosine similarity alignment loss
        cos_sim = F.cosine_similarity(outputs['vision_universal'], outputs['language_universal'])
        losses['alignment'] = 1 - cos_sim.mean()
    else:
        losses['alignment'] = torch.tensor(0.0)
    
    # Total loss
    losses['total'] = losses['recon_total'] + 0.1 * losses['alignment']
    
    return losses


def evaluate_retrieval(embeddings1, embeddings2, labels):
    """Evaluate retrieval metrics"""
    embeddings1 = F.normalize(torch.from_numpy(embeddings1), dim=1).numpy()
    embeddings2 = F.normalize(torch.from_numpy(embeddings2), dim=1).numpy()
    
    similarities = np.dot(embeddings1, embeddings2.T)
    
    # Instance-level retrieval
    instance_r1_12 = 0
    instance_r1_21 = 0
    
    # Species-level retrieval
    species_r1_12 = 0
    species_r1_21 = 0
    
    for i in range(len(embeddings1)):
        # 1 → 2 retrieval
        sorted_indices = np.argsort(similarities[i])[::-1]
        if sorted_indices[0] == i:
            instance_r1_12 += 1
        
        # Find first same species
        for idx in sorted_indices:
            if labels[idx] == labels[i]:
                species_r1_12 += 1
                break
        
        # 2 → 1 retrieval
        sorted_indices = np.argsort(similarities[:, i])[::-1]
        if sorted_indices[0] == i:
            instance_r1_21 += 1
            
        # Find first same species
        for idx in sorted_indices:
            if labels[idx] == labels[i]:
                species_r1_21 += 1
                break
    
    n = len(embeddings1)
    return {
        'instance_r1_12': instance_r1_12 / n,
        'instance_r1_21': instance_r1_21 / n,
        'species_r1_12': species_r1_12 / n,
        'species_r1_21': species_r1_21 / n,
    }


def create_umap_visualization(embeddings, labels, species_names, save_path, title="UMAP"):
    """Create UMAP visualization"""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=species_names[label][:20], s=30, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DeepEarth MLP U-Net Training')
    parser.add_argument('--mode', choices=['language', 'vision', 'both'], default='both',
                       help='Training mode')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save-dir', type=str, default='checkpoints_dashboard', help='Save directory')
    parser.add_argument('--mask-ratio', type=float, default=0.5, help='Language masking ratio')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available() and args.device == 'cuda':
        args.device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    set_all_seeds(args.seed)
    
    print("🌍 DeepEarth MLP U-Net Training (Dashboard Integrated)")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    
    # Initialize cache
    print("\n📊 Loading data cache...")
    try:
        # Change to dashboard directory for cache initialization
        import os
        original_cwd = os.getcwd()
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        os.chdir(dashboard_dir)
        
        cache = UnifiedDataCache("dataset_config.json")
        available_ids = get_available_observation_ids(cache)
        
        os.chdir(original_cwd)
        
        print(f"Total available observations: {len(available_ids)}")
        
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        return 1
    
    # Create train/test split
    random.shuffle(available_ids)
    available_ids = available_ids[:args.max_samples]
    
    split_idx = int(0.8 * len(available_ids))
    train_ids = available_ids[:split_idx]
    test_ids = available_ids[split_idx:]
    
    print(f"\nCreating datasets...")
    print(f"  Train: {len(train_ids)} observations")
    print(f"  Test: {len(test_ids)} observations")
    
    # Create datasets with consistent species mapping
    # First, sample both to get common species
    sample_train = DeepEarthMLPDataset(train_ids[:200], cache, args.mode, 'cpu')
    train_species = set(sample_train.species_to_idx.keys())
    
    sample_test = DeepEarthMLPDataset(test_ids[:50], cache, args.mode, 'cpu')
    test_species = set(sample_test.species_to_idx.keys())
    
    common_species = train_species & test_species
    if len(common_species) < 5:
        logger.warning(f"Only {len(common_species)} common species between train/test. Using all species.")
        common_species = train_species | test_species
    
    species_mapping = {species: idx for idx, species in enumerate(sorted(common_species))}
    species_names = {idx: species for species, idx in species_mapping.items()}
    
    # Create full datasets with consistent mapping
    train_dataset = DeepEarthMLPDataset(train_ids, cache, args.mode, args.device, species_mapping)
    test_dataset = DeepEarthMLPDataset(test_ids, cache, args.mode, args.device, species_mapping)
    
    print(f"\nDataset stats:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of species: {train_dataset.num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = SimpleMLPUNet().to(args.device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # CSV logging
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'recon_loss', 'align_loss', 
                        'species_r1_v2l', 'species_r1_l2v'])
    
    # Training loop
    print(f"\n🚀 Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'alignment': 0}
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in progress:
            # Generate language mask
            if args.mode in ['language', 'both'] and args.mask_ratio > 0:
                language_mask = torch.rand_like(batch['language_embedding'], dtype=torch.float) < args.mask_ratio
            else:
                language_mask = None
            
            # Forward pass
            outputs = model(
                batch['vision_embedding'] if args.mode in ['vision', 'both'] else torch.zeros(len(batch['species_idx']), 8, 24, 24, 1408, device=args.device),
                batch['language_embedding'] if args.mode in ['language', 'both'] else torch.zeros(len(batch['species_idx']), 7168, device=args.device),
                mask_language=language_mask,
                mode=args.mode
            )
            
            losses = compute_losses(outputs)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['recon'] += losses['recon_total'].item()
            epoch_losses['alignment'] += losses['alignment'].item()
            
            progress.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon_total'].item():.4f}"
            })
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        scheduler.step()
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            
            all_vision_emb = []
            all_language_emb = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(
                        batch['vision_embedding'] if args.mode in ['vision', 'both'] else torch.zeros(len(batch['species_idx']), 8, 24, 24, 1408, device=args.device),
                        batch['language_embedding'] if args.mode in ['language', 'both'] else torch.zeros(len(batch['species_idx']), 7168, device=args.device),
                        mode=args.mode
                    )
                    
                    all_vision_emb.append(outputs['vision_universal'].cpu().numpy())
                    all_language_emb.append(outputs['language_universal'].cpu().numpy())
                    all_labels.append(batch['species_idx'].cpu().numpy())
            
            all_vision_emb = np.vstack(all_vision_emb)
            all_language_emb = np.vstack(all_language_emb)
            all_labels = np.hstack(all_labels)
            
            # Retrieval evaluation
            retrieval_results = evaluate_retrieval(all_vision_emb, all_language_emb, all_labels)
            
            print(f"\nEpoch {epoch+1} Evaluation:")
            print(f"  Loss: {epoch_losses['total']:.4f}")
            print(f"  Species R@1: V→L={retrieval_results['species_r1_12']:.3f}, "
                  f"L→V={retrieval_results['species_r1_21']:.3f}")
            
            # Write to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, epoch_losses['total'], epoch_losses['recon'], 
                               epoch_losses['alignment'], retrieval_results['species_r1_12'], 
                               retrieval_results['species_r1_21']])
            
            # UMAP visualization
            if (epoch + 1) % 10 == 0:
                umap_path = save_dir / f'umap_epoch_{epoch+1}.png'
                create_umap_visualization(
                    all_vision_emb, all_labels, species_names,
                    umap_path, f"UMAP - Epoch {epoch+1}"
                )
                print(f"  Saved UMAP to {umap_path}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'args': vars(args)
            }
            torch.save(checkpoint, save_dir / 'latest_checkpoint.pth')
    
    print("\n✅ Training complete!")
    print(f"Results saved to: {args.save_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
