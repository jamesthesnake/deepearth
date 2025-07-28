#!/usr/bin/env python3
"""
DeepEarth Multimodal MLP U-Net Training

Combines V-JEPA 2 vision embeddings with DeepSeek language embeddings
using a U-Net architecture with skip connections.

Usage:
    python train_multimodal_unet.py --epochs 50 --batch-size 64
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
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Add dashboard to path for data access
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Import your existing infrastructure
from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

# Import base components from existing train_classifier.py
sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import DeepEarthDataset, load_train_test_split, create_training_visualization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalMLPUNet(nn.Module):
    """
    🔷 Multimodal MLP U-Net for V-JEPA 2 + DeepSeek fusion
    
    Architecture:
    - Separate encoders for vision and language
    - Bottleneck fusion layer
    - Decoders with skip connections
    - Classification from bottleneck
    """
    
    def __init__(self, num_classes: int, vision_dim: int = 1408, language_dim: int = 7168,
                 hidden_dims: List[int] = [2048, 1024, 512, 256]):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dims = hidden_dims
        
        # Vision encoder layers
        self.vision_encoders = nn.ModuleList()
        dims = [vision_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.vision_encoders.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # Language encoder layers
        self.language_encoders = nn.ModuleList()
        dims = [language_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.language_encoders.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.1)
            ))
        
        # Bottleneck fusion
        bottleneck_dim = hidden_dims[-1]
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )
        
        # Classifier from bottleneck
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_dim // 2, num_classes)
        )
        
        # Vision decoder layers with skip connections
        self.vision_decoders = nn.ModuleList()
        dims_reversed = hidden_dims[::-1] + [vision_dim]
        for i in range(len(dims_reversed) - 1):
            # Account for skip connections (double size except first layer)
            in_dim = dims_reversed[i] if i == 0 else dims_reversed[i] * 2
            out_dim = dims_reversed[i+1]
            
            if i < len(dims_reversed) - 2:  # Hidden layers
                self.vision_decoders.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ))
            else:  # Output layer
                self.vision_decoders.append(nn.Linear(in_dim, out_dim))
        
        # Language decoder layers with skip connections
        self.language_decoders = nn.ModuleList()
        dims_reversed = hidden_dims[::-1] + [language_dim]
        for i in range(len(dims_reversed) - 1):
            in_dim = dims_reversed[i] if i == 0 else dims_reversed[i] * 2
            out_dim = dims_reversed[i+1]
            
            if i < len(dims_reversed) - 2:
                self.language_decoders.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ))
            else:
                self.language_decoders.append(nn.Linear(in_dim, out_dim))
    
    def forward(self, vision_emb, language_emb):
        # Store skip connections
        vision_skips = []
        language_skips = []
        
        # Vision encoding
        v = vision_emb
        for i, encoder in enumerate(self.vision_encoders):
            v = encoder(v)
            if i < len(self.vision_encoders) - 1:  # Don't save last layer
                vision_skips.append(v)
        
        # Language encoding
        l = language_emb
        for i, encoder in enumerate(self.language_encoders):
            l = encoder(l)
            if i < len(self.language_encoders) - 1:
                language_skips.append(l)
        
        # Bottleneck fusion
        fused = torch.cat([v, l], dim=1)
        bottleneck = self.fusion(fused)
        
        # Classification
        logits = self.classifier(bottleneck)
        
        # Vision decoding with skip connections
        v_dec = bottleneck
        for i, decoder in enumerate(self.vision_decoders):
            if i > 0 and i <= len(vision_skips):
                skip_idx = -(i)  # Reverse order
                v_dec = torch.cat([v_dec, vision_skips[skip_idx]], dim=1)
            v_dec = decoder(v_dec)
        
        # Language decoding with skip connections
        l_dec = bottleneck
        for i, decoder in enumerate(self.language_decoders):
            if i > 0 and i <= len(language_skips):
                skip_idx = -(i)
                l_dec = torch.cat([l_dec, language_skips[skip_idx]], dim=1)
            l_dec = decoder(l_dec)
        
        return {
            'logits': logits,
            'vision_reconstruction': v_dec,
            'language_reconstruction': l_dec,
            'bottleneck': bottleneck,
            'vision_encoded': v,
            'language_encoded': l
        }


class MultimodalDataset(DeepEarthDataset):
    """Extended dataset for multimodal training"""
    
    def __init__(self, observation_ids: List[str], cache, device: str = 'cpu', 
                 species_mapping: Optional[Dict] = None):
        # Initialize with both modalities
        super().__init__(observation_ids, cache, mode='both', device=device, 
                        species_mapping=species_mapping)
    
    def __getitem__(self, idx):
        """Get both vision and language embeddings"""
        # For vision: average pool over temporal and spatial dimensions
        vision_emb = self.vision_embeddings[idx]  # Shape: (8, 24, 24, 1408)
        vision_pooled = vision_emb.mean(dim=(0, 1, 2))  # Shape: (1408,)
        
        return {
            'vision_embedding': vision_pooled,
            'language_embedding': self.language_embeddings[idx],
            'species_label': self.species_labels[idx]
        }


def compute_loss(outputs, batch, lambda_rec=0.1):
    """Compute combined classification and reconstruction loss"""
    # Classification loss
    cls_loss = F.cross_entropy(outputs['logits'], batch['species_label'])
    
    # Reconstruction losses (normalized by dimension)
    vision_rec_loss = F.mse_loss(outputs['vision_reconstruction'], 
                                 batch['vision_embedding']) / 1408
    language_rec_loss = F.mse_loss(outputs['language_reconstruction'], 
                                   batch['language_embedding']) / 7168
    
    rec_loss = (vision_rec_loss + language_rec_loss) / 2
    
    # Combined loss
    total_loss = cls_loss + lambda_rec * rec_loss
    
    return {
        'total': total_loss,
        'classification': cls_loss,
        'reconstruction': rec_loss,
        'vision_rec': vision_rec_loss,
        'language_rec': language_rec_loss
    }


def train_epoch(model, dataloader, optimizer, lambda_rec=0.1):
    """Train for one epoch"""
    model.train()
    
    metrics = defaultdict(float)
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch['vision_embedding'], batch['language_embedding'])
        losses = compute_loss(outputs, batch, lambda_rec)
        
        # Backward pass
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        for k, v in losses.items():
            metrics[k] += v.item()
        
        # Accuracy
        _, predicted = torch.max(outputs['logits'], 1)
        total_correct += (predicted == batch['species_label']).sum().item()
        total_samples += batch['species_label'].size(0)
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                       f"Loss={losses['total'].item():.4f}, "
                       f"Cls={losses['classification'].item():.4f}, "
                       f"Rec={losses['reconstruction'].item():.4f}")
    
    # Average metrics
    num_batches = len(dataloader)
    for k in metrics:
        metrics[k] /= num_batches
    
    accuracy = 100.0 * total_correct / total_samples
    return metrics['total'], accuracy, metrics


@torch.no_grad()
def evaluate(model, dataloader, lambda_rec=0.1):
    """Evaluate model"""
    model.eval()
    
    metrics = defaultdict(float)
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        outputs = model(batch['vision_embedding'], batch['language_embedding'])
        losses = compute_loss(outputs, batch, lambda_rec)
        
        for k, v in losses.items():
            metrics[k] += v.item()
        
        _, predicted = torch.max(outputs['logits'], 1)
        total_correct += (predicted == batch['species_label']).sum().item()
        total_samples += batch['species_label'].size(0)
    
    # Average metrics
    num_batches = len(dataloader)
    for k in metrics:
        metrics[k] /= num_batches
    
    accuracy = 100.0 * total_correct / total_samples
    return metrics['total'], accuracy, metrics


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='DeepEarth Multimodal MLP U-Net Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda-rec', type=float, default=0.1, help='Reconstruction loss weight')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda/cpu/auto')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[2048, 1024, 512, 256],
                       help='Hidden dimensions for U-Net')
    parser.add_argument('--config', type=str, help='Path to train/test split config')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("🔷 DeepEarth Multimodal MLP U-Net Training")
    print(f"Device: {device}")
    print(f"Architecture: {args.hidden_dims}")
    print(f"λ_rec: {args.lambda_rec}")
    
    # Initialize cache and load splits
    try:
        # Save original directory
        original_cwd = Path.cwd()
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        
        # Auto-detect config if not provided (before changing directory)
        if not args.config:
            config_path = Path(__file__).parent / "config" / "central_florida_split.json"
            args.config = str(config_path)
        
        # Load train/test split first (from current directory)
        train_ids, test_ids = load_train_test_split(args.config)
        
        # Now change to dashboard directory for cache initialization
        import os
        os.chdir(dashboard_dir)
        
        # Initialize cache (this needs to be in dashboard directory)
        cache = UnifiedDataCache("dataset_config.json")
        
        # Stay in dashboard directory for data loading
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    print(f"\n📊 Loading multimodal data...")
    print(f"Train: {len(train_ids)} observations")
    print(f"Test: {len(test_ids)} observations")
    
    # Create datasets (staying in dashboard directory)
    try:
        train_dataset = MultimodalDataset(train_ids, cache, device)
        test_dataset = MultimodalDataset(test_ids, cache, device, 
                                        species_mapping=train_dataset.species_mapping)
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        # Try to return to original directory
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return 1
    
    # Filter to common species
    train_species = set([train_dataset.idx_to_species[i] for i in range(train_dataset.num_classes)])
    test_species_indices = [i for i in range(len(test_dataset)) 
                           if test_dataset.idx_to_species[test_dataset.species_labels[i].item()] in train_species]
    
    if len(test_species_indices) < len(test_dataset):
        print(f"Filtering test set to {len(test_species_indices)} samples with species in training set")
        test_dataset = torch.utils.data.Subset(test_dataset, test_species_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nDataset ready:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Species: {train_dataset.num_classes}")
    
    # Initialize model
    model = MultimodalMLPUNet(
        num_classes=train_dataset.num_classes,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    print(f"\n🧠 Model initialized")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, args.lambda_rec
        )
        scheduler.step()
        
        # Evaluate
        test_loss, test_acc, test_metrics = evaluate(
            model, test_loader, args.lambda_rec
        )
        
        # Record
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"    Cls: {train_metrics['classification']:.4f}, "
              f"Rec: {train_metrics['reconstruction']:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.1f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_accuracy': test_acc,
                'args': vars(args),
                'species_mapping': train_dataset.species_to_idx
            }
            torch.save(checkpoint, 'best_multimodal_unet.pth')
            print(f"  💾 New best model saved! ({test_acc:.1f}%)")
    
    # Final results
    print(f"\n✅ Training complete!")
    print(f"Best test accuracy: {best_test_acc:.1f}%")
    
    # Create visualization
    viz_path = create_training_visualization(
        train_losses, train_accuracies, test_accuracies, 'multimodal'
    )
    print(f"📊 Visualization saved to: {viz_path}")
    
    # Return to original directory
    if 'original_cwd' in locals():
        os.chdir(original_cwd)
    
    return 0


if __name__ == "__main__":
    exit(main())
