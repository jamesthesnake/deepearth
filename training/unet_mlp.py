#!/usr/bin/env python3
"""
Multimodal Autoencoder for DeepEarth with Hierarchical U-Net
- Hierarchical U-Net for multimodal fusion
- MLP decoders integrated into U-Net architecture
- Support for simultaneous vision and language masking
- Universal 1x2048 embedding space
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
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import umap
from einops import rearrange
from einops.layers.torch import Rearrange

# Keep all your existing imports and utility functions
def create_alignment_visualization(model, loader, device, epoch, save_dir):
    """Create UMAP visualization of vision-language alignment"""
    model.eval()
    
    all_vision_universal = []
    all_language_universal = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            labels = batch['species_label'].to(device)
            
            # Pass labels for ArcFace compatibility
            if hasattr(model, 'use_arcface') and model.use_arcface:
                outputs = model(vision, language, labels=labels)
            else:
                outputs = model(vision, language)
            all_vision_universal.append(outputs['vision_universal'].cpu())
            all_language_universal.append(outputs['language_universal'].cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate
    vision_universal = torch.cat(all_vision_universal, dim=0).numpy()
    language_universal = torch.cat(all_language_universal, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Normalize for cosine similarity
    vision_norm = F.normalize(torch.from_numpy(vision_universal), p=2, dim=1).numpy()
    language_norm = F.normalize(torch.from_numpy(language_universal), p=2, dim=1).numpy()
    
    # Compute alignment metrics
    n_samples = len(labels)
    diagonal_sim = np.sum(vision_norm * language_norm, axis=1)  # Paired similarity
    avg_alignment = diagonal_sim.mean()
    
    # Cross-modal retrieval
    v2l_correct = 0
    for i in range(n_samples):
        sims = np.dot(vision_norm[i], language_norm.T)
        if labels[sims.argmax()] == labels[i]:
            v2l_correct += 1
    v2l_acc = v2l_correct / n_samples
    
    # UMAP visualization
    combined = np.vstack([vision_universal, language_universal])
    combined_labels = np.hstack([labels, labels])
    modality = ['Vision'] * n_samples + ['Language'] * n_samples
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embeddings_2d = reducer.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # By modality
    for mod in ['Vision', 'Language']:
        mask = np.array(modality) == mod
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   label=mod, alpha=0.6, s=30)
    plt.title(f'Universal Space by Modality - Epoch {epoch}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Connected pairs
    plt.scatter(embeddings_2d[:n_samples, 0], embeddings_2d[:n_samples, 1], 
               c=labels, cmap='tab20', alpha=0.8, s=50, marker='o')
    plt.scatter(embeddings_2d[n_samples:, 0], embeddings_2d[n_samples:, 1], 
               c=labels, cmap='tab20', alpha=0.8, s=50, marker='^')
    
    # Draw connections
    for i in range(min(n_samples, 50)):  # Limit lines for clarity
        plt.plot([embeddings_2d[i, 0], embeddings_2d[n_samples + i, 0]], 
                [embeddings_2d[i, 1], embeddings_2d[n_samples + i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    plt.title(f'V-L Alignment - Avg Sim: {avg_alignment:.3f}, V→L R@1: {v2l_acc:.2%}')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'alignment_epoch_{epoch}.png', dpi=150)
    plt.close()
    
    logger.info(f"Alignment metrics - Avg similarity: {avg_alignment:.3f}, V→L retrieval: {v2l_acc:.2%}")
    
    return avg_alignment, v2l_acc

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Suppress verbose logging
logging.getLogger('services.training_data').setLevel(logging.WARNING)
logging.getLogger('mmap_embedding_loader').setLevel(logging.WARNING)
logging.getLogger('huggingface_data_loader').setLevel(logging.WARNING)
logging.getLogger('data_cache').setLevel(logging.WARNING)

from services.training_data import get_training_batch
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Optimized dataset that pre-computes species labels for fast sampler access
class EfficientMultimodalDataset(Dataset):
    """Memory-efficient dataset that loads data on-demand with pre-computed labels"""
    
    def __init__(self, observation_ids: List[str], cache, device='cpu', species_mapping=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.device = device
        
        if species_mapping is not None:
            self.species_to_idx = species_mapping
            self.num_classes = len(species_mapping)
            logger.info(f"Using provided species mapping with {self.num_classes} classes")
        else:
            # Build species mapping from ALL observations
            logger.info("Building species mapping from all observations...")
            all_species = set()
            
            # Sample in batches to find all species
            batch_size = 100
            for i in range(0, len(self.observation_ids), batch_size):
                batch_ids = self.observation_ids[i:i+batch_size]
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=False,
                    include_language=True,
                    device='cpu'
                )
                all_species.update(batch_data['species'])
                
                if i % 1000 == 0:
                    logger.info(f"  Processed {i}/{len(self.observation_ids)} observations, "
                               f"found {len(all_species)} species so far")
            
            self.species_to_idx = {s: i for i, s in enumerate(sorted(all_species))}
            self.num_classes = len(self.species_to_idx)
            logger.info(f"Found {self.num_classes} unique species in dataset")
        
        # Pre-compute all species labels for fast access (needed for BalancedSpeciesSampler)
        logger.info("Pre-computing species labels for fast sampler access...")
        self.species_labels = []
        self.valid_indices = []
        
        batch_size = 100
        for i in range(0, len(observation_ids), batch_size):
            if i % 1000 == 0:
                logger.info(f"  Pre-computing labels: {i}/{len(observation_ids)}...")
            
            batch_ids = observation_ids[i:i+batch_size]
            batch_data = get_training_batch(
                self.cache,
                batch_ids,
                include_vision=False,
                include_language=True,
                device='cpu'
            )
            
            for j, species in enumerate(batch_data['species']):
                if species in self.species_to_idx:
                    self.species_labels.append(self.species_to_idx[species])
                    self.valid_indices.append(i + j)
        
        logger.info(f"Pre-computed {len(self.species_labels)} valid labels")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def get_species_label(self, idx):
        """Fast access to species label without loading embeddings"""
        return self.species_labels[idx]
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        obs_id = self.observation_ids[actual_idx]
        
        try:
            batch_data = get_training_batch(
                self.cache,
                [obs_id],
                include_vision=True,
                include_language=True,
                device='cpu'
            )
            
            species = batch_data['species'][0]
            
            if species not in self.species_to_idx:
                logger.warning(f"Species '{species}' not in mapping, skipping")
                return self.__getitem__(np.random.randint(0, len(self)))
            
            vision_emb = batch_data['vision_embeddings'][0]  # (8, 24, 24, 1408)
            language_emb = batch_data['language_embeddings'][0]  # (7168,)
            
            # Pool vision embedding
            vision_pooled = vision_emb.mean(dim=(0, 1, 2))  # (1408,)
            
            return {
                'vision_embedding': vision_pooled,
                'language_embedding': language_emb,
                'species_label': torch.tensor(self.species_labels[idx], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error loading observation {obs_id}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


class MultimodalAutoencoder(nn.Module):
    """
    Original Multimodal Autoencoder with separate MLP decoders
    Following meeting notes:
    - Vision encoder: 1408 → 2048
    - Language encoder: 7168 → 2048  
    - Separate 2-layer MLP decoders for each modality
    - Configurable masking per modality
    """
    
    def __init__(self, num_classes, vision_dim=1408, language_dim=7168, 
                 universal_dim=2048, hidden_dim=512, dropout_rate=0.2):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        
        # Modality encoders to universal dimension
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion at universal dimension
        self.fusion = nn.Sequential(
            nn.Linear(universal_dim * 2, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier from universal embedding
        self.classifier = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Vision decoder - 2 layer MLP as specified
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, vision_dim)
        )
        
        # Language decoder - 2 layer MLP as specified  
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, language_dim)
        )
        
        # Projection heads for contrastive learning
        self.vision_projection = ProjectionHead(universal_dim, hidden_dim=1024, output_dim=256)
        self.language_projection = ProjectionHead(universal_dim, hidden_dim=1024, output_dim=256)
    
    def forward(self, vision, language, mask_vision=False, mask_language=False, 
                vision_mask_ratio=0.5, language_mask_ratio=0.5, mask_type='feature', labels=None):
        """
        Forward pass with configurable masking per modality
        
        Args:
            vision: Vision embeddings (batch, 1408)
            language: Language embeddings (batch, 7168)
            mask_vision: Whether to mask vision modality
            mask_language: Whether to mask language modality
            vision_mask_ratio: Ratio of vision features to mask
            language_mask_ratio: Ratio of language features to mask
            mask_type: 'feature' (mask dimensions) or 'sample' (mask entire embeddings)
        """
        batch_size = vision.shape[0]
        device = vision.device
        
        # Store originals for reconstruction loss
        original_vision = vision.clone()
        original_language = language.clone()
        
        # Apply masking if requested
        vision_mask = None
        language_mask = None
        
        if mask_vision and self.training:
            if mask_type == 'sample':
                # Mask entire samples
                vision_mask = torch.rand(batch_size, device=device) < vision_mask_ratio
                vision = vision * (~vision_mask).float().unsqueeze(1)
            else:
                # Feature-level masking
                vision_mask = torch.rand(batch_size, self.vision_dim, device=device) < vision_mask_ratio
                vision = vision * (~vision_mask).float()
            
        if mask_language and self.training:
            if mask_type == 'sample':
                # Mask entire samples (50% of language embeddings completely zeroed)
                language_mask = torch.rand(batch_size, device=device) < language_mask_ratio
                language = language * (~language_mask).float().unsqueeze(1)
            else:
                # Feature-level masking (50% of dimensions zeroed for all samples)
                language_mask = torch.rand(batch_size, self.language_dim, device=device) < language_mask_ratio
                language = language * (~language_mask).float()
        
        # Encode to universal space
        vision_universal = self.vision_encoder(vision)      # 1408 → 2048
        language_universal = self.language_encoder(language)  # 7168 → 2048
        
        # Fusion
        fused = torch.cat([vision_universal, language_universal], dim=1)  # 4096
        bottleneck = self.fusion(fused)  # 4096 → 2048 (universal embedding)
        
        # Classification
        logits = self.classifier(bottleneck)
        
        # Decode from universal embedding
        vision_recon = self.vision_decoder(bottleneck)      # 2048 → 1408
        language_recon = self.language_decoder(bottleneck)  # 2048 → 7168
        
        # Get projections for contrastive learning
        vision_proj, vision_temp = self.vision_projection(vision_universal)
        language_proj, language_temp = self.language_projection(language_universal)
        
        # Average temperatures
        temperature = (vision_temp + language_temp) / 2
        
        return {
            'logits': logits,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'bottleneck': bottleneck,  # The 1x2048 universal embedding
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'vision_proj': vision_proj,
            'language_proj': language_proj,
            'temperature': temperature,
            'original_vision': original_vision,
            'original_language': original_language,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }


class ProjectionHead(nn.Module):
    """Enhanced projection head with dual projection and learnable temperature"""
    def __init__(self, input_dim, hidden_dim=1024, output_dim=256, use_bn=True):
        super().__init__()
        
        # Two-layer projection with BN and GELU
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.gelu = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        
        # Learnable temperature (log scale for stability)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        x = self.proj1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.proj2(x)
        return x, self.logit_scale.exp()


class ArcFaceLoss(nn.Module):
    """ArcFace loss for better inter-class separation"""
    def __init__(self, in_features, out_features, s=30.0, m=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scale
        self.m = m  # Margin
        
        # Weight normalized
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(features, weights)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Convert to angle
        theta = torch.acos(cos_theta)
        
        # Add margin to target angle
        target_logits = torch.cos(theta + self.m)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Combine
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        
        return output


def compute_variance_penalty(embeddings):
    """Soft hinge to encourage variance > 0.5 per dimension"""
    var = torch.var(embeddings, dim=0)
    # Push variance above 0.5 instead of strictly to 1
    penalty = torch.relu(0.5 - var).mean()
    return penalty


def compute_center_penalty(vision_emb, language_emb):
    """Keep modality centers aligned"""
    vision_center = vision_emb.mean(dim=0)
    language_center = language_emb.mean(dim=0)
    return F.mse_loss(vision_center, language_center)


def temperature_schedule(epoch, total_epochs, initial_temp=0.2, final_temp=0.07):
    """Cosine warmup for temperature"""
    warmup_epochs = total_epochs // 3
    if epoch < warmup_epochs:
        # Cosine warmup from initial to final
        progress = epoch / warmup_epochs
        return final_temp + (initial_temp - final_temp) * (1 + np.cos(np.pi * progress)) / 2
    else:
        return final_temp


class BalancedSpeciesSampler(torch.utils.data.Sampler):
    """Balanced sampling with cap on samples per species - optimized with pre-computed labels"""
    def __init__(self, dataset, max_per_species_factor=3.0):  # Changed from 1.5 to 3.0
        self.dataset = dataset
        self.num_samples = len(dataset)
        
        # Build species to indices mapping using fast label access
        self.species_to_indices = {}
        
        logger.info(f"Building species index for {self.num_samples} samples...")
        
        if hasattr(dataset, 'get_species_label'):
            # Use fast label access (much faster!)
            for idx in range(len(dataset)):
                if idx % 1000 == 0:
                    print(f"\r  Processed {idx}/{self.num_samples} samples...", end='', flush=True)
                
                species = dataset.get_species_label(idx)
                if species not in self.species_to_indices:
                    self.species_to_indices[species] = []
                self.species_to_indices[species].append(idx)
            print()
        else:
            # Fallback to original method (slow - loads all embeddings)
            logger.warning("Dataset doesn't support fast label access, using slow method...")
            for idx in range(len(dataset)):
                if idx % 100 == 0:
                    print(f"\r  Processed {idx}/{self.num_samples} samples...", end='', flush=True)
                
                species = dataset[idx]['species_label'].item()
                if species not in self.species_to_indices:
                    self.species_to_indices[species] = []
                self.species_to_indices[species].append(idx)
            print()
        
        # Calculate median and cap
        species_counts = [len(indices) for indices in self.species_to_indices.values()]
        median_count = np.median(species_counts)
        self.max_per_species = int(median_count * max_per_species_factor)
        
        logger.info(f"Found {len(self.species_to_indices)} species")
        logger.info(f"Species distribution: median={median_count:.1f}, max_per_species={self.max_per_species}")
        
    def __iter__(self):
        # Sample balanced batches
        indices = []
        species_list = list(self.species_to_indices.keys())
        
        while len(indices) < self.num_samples:
            # Shuffle species order
            np.random.shuffle(species_list)
            
            for species in species_list:
                if len(indices) >= self.num_samples:
                    break
                    
                species_indices = self.species_to_indices[species]
                # Cap samples per species
                n_samples = min(1, len(species_indices), self.max_per_species)
                sampled = np.random.choice(species_indices, n_samples, replace=False)
                indices.extend(sampled)
        
        return iter(indices[:self.num_samples])
    
    def __len__(self):
        return self.num_samples


class CrossModalSpatialAttention(nn.Module):
    """Spatial attention mechanism for cross-modal interaction in U-Net bottleneck"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Separate pathways for vision and language channels
        self.vision_query = nn.Conv2d(channels // 2, channels // 4, 1)
        self.vision_key = nn.Conv2d(channels // 2, channels // 4, 1)
        self.vision_value = nn.Conv2d(channels // 2, channels // 2, 1)
        
        self.language_query = nn.Conv2d(channels // 2, channels // 4, 1)
        self.language_key = nn.Conv2d(channels // 2, channels // 4, 1)
        self.language_value = nn.Conv2d(channels // 2, channels // 2, 1)
        
        self.output_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(16, channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split channels (assuming vision and language are concatenated)
        vision_feat = x[:, :C//2]
        language_feat = x[:, C//2:]
        
        # Cross attention: vision attending to language
        v_q = self.vision_query(vision_feat).view(B, -1, H * W).transpose(1, 2)
        l_k = self.language_key(language_feat).view(B, -1, H * W)
        l_v = self.language_value(language_feat).view(B, -1, H * W).transpose(1, 2)
        
        v2l_attention = F.softmax(torch.bmm(v_q, l_k) / np.sqrt(C // 4), dim=-1)
        v2l_out = torch.bmm(v2l_attention, l_v).transpose(1, 2).view(B, C//2, H, W)
        
        # Cross attention: language attending to vision
        l_q = self.language_query(language_feat).view(B, -1, H * W).transpose(1, 2)
        v_k = self.vision_key(vision_feat).view(B, -1, H * W)
        v_v = self.vision_value(vision_feat).view(B, -1, H * W).transpose(1, 2)
        
        l2v_attention = F.softmax(torch.bmm(l_q, v_k) / np.sqrt(C // 4), dim=-1)
        l2v_out = torch.bmm(l2v_attention, v_v).transpose(1, 2).view(B, C//2, H, W)
        
        # Combine with residual
        out = torch.cat([vision_feat + v2l_out, language_feat + l2v_out], dim=1)
        out = self.output_conv(out)
        out = self.norm(out)
        
        return out + x  # Final residual


class MultimodalAutoencoderWithHierarchicalUNet(nn.Module):
    """
    Multimodal Autoencoder with Hierarchical U-Net
    
    Key Architecture:
    1. Modality encoders project to universal space (2048)
    2. Stack universal embeddings: (batch, n_modalities, 2048)
    3. U-Net processes universal embeddings with dynamic masking
    4. Separate decoders reconstruct original modalities from U-Net output
    """
    
    def __init__(self, num_classes, vision_dim=1408, language_dim=7168, 
                 universal_dim=2048, hidden_dim=512, dropout_rate=0.2,
                 unet_features=[256, 512, 1024, 2048],
                 use_spatial_attention=True):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        self.n_modalities = 2  # vision + language
        
        # Learnable mask tokens for each modality in universal space
        self.vision_mask_token = nn.Parameter(torch.randn(1, 1, universal_dim))
        self.language_mask_token = nn.Parameter(torch.randn(1, 1, universal_dim))
        
        # Step 1: Modality encoders to universal dimension
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Step 2: U-Net to process stacked universal embeddings
        # Input shape: (batch, n_modalities, universal_dim)
        self.unet = HierarchicalUNet(
            in_channels=self.n_modalities,
            out_channels=self.n_modalities,
            features=unet_features,
            embedding_dim=universal_dim,
            use_attention=use_spatial_attention
        )
        
        # Step 3: Fusion of U-Net output modalities
        self.fusion = nn.Sequential(
            nn.Linear(universal_dim * self.n_modalities, universal_dim),
            nn.LayerNorm(universal_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Step 4: Classifier from fused universal embedding
        self.classifier = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Step 5: Modality-specific decoders from universal to original dimensions
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, vision_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, language_dim)
        )
        
        # Step 6: Projection heads for contrastive learning (FIXED - WAS MISSING)
        self.vision_projection = ProjectionHead(universal_dim, hidden_dim=1024, output_dim=256)
        self.language_projection = ProjectionHead(universal_dim, hidden_dim=1024, output_dim=256)
    
    def forward(self, vision, language, mask_vision=False, mask_language=False, 
                vision_mask_ratio=0.5, language_mask_ratio=0.5, mask_type='feature', labels=None):
        """
        Forward pass following the specified architecture:
        1. Encode modalities to universal space
        2. Stack universal embeddings
        3. Process with U-Net (with optional masking)
        4. Decode back to original modalities
        """
        batch_size = vision.shape[0]
        device = vision.device
        
        # Store originals for reconstruction loss
        original_vision = vision.clone()
        original_language = language.clone()
        
        # Step 1: Encode to universal dimension
        vision_universal = self.vision_encoder(vision)        # (batch, 2048)
        language_universal = self.language_encoder(language)  # (batch, 2048)
        
        # Step 2: Stack modalities for U-Net input
        # Shape: (batch, n_modalities, universal_dim)
        stacked_universal = torch.stack([vision_universal, language_universal], dim=1)  # (batch, 2, 2048)
        
        # Step 3: Apply masking at universal level if requested
        vision_mask = None
        language_mask = None
        
        if self.training:
            if mask_vision:
                if mask_type == 'token':
                    # Mask entire modality token
                    vision_mask = torch.rand(batch_size, device=device) < vision_mask_ratio
                    stacked_universal[:, 0] = torch.where(
                        vision_mask.unsqueeze(1),
                        self.vision_mask_token.expand(batch_size, -1, -1).squeeze(1),
                        stacked_universal[:, 0]
                    )
                else:
                    # Feature-level masking in universal space
                    vision_mask = torch.rand(batch_size, self.universal_dim, device=device) < vision_mask_ratio
                    stacked_universal[:, 0] = stacked_universal[:, 0] * (~vision_mask).float()
                    
            if mask_language:
                if mask_type == 'token':
                    # Mask entire modality token
                    language_mask = torch.rand(batch_size, device=device) < language_mask_ratio
                    stacked_universal[:, 1] = torch.where(
                        language_mask.unsqueeze(1),
                        self.language_mask_token.expand(batch_size, -1, -1).squeeze(1),
                        stacked_universal[:, 1]
                    )
                else:
                    # Feature-level masking in universal space
                    language_mask = torch.rand(batch_size, self.universal_dim, device=device) < language_mask_ratio
                    stacked_universal[:, 1] = stacked_universal[:, 1] * (~language_mask).float()
        
        # Step 4: Process through U-Net
        # U-Net expects (batch, channels, height, width) so we treat universal_dim as spatial
        unet_input = stacked_universal  # (batch, 2, 2048)
        unet_output = self.unet(unet_input)  # (batch, 2, 2048)
        
        # Step 5: Extract reconstructed universal embeddings
        vision_universal_recon = unet_output[:, 0]      # (batch, 2048)
        language_universal_recon = unet_output[:, 1]    # (batch, 2048)
        
        # Step 6: Fusion for classification
        fused = torch.cat([vision_universal_recon, language_universal_recon], dim=1)
        bottleneck = self.fusion(fused)  # (batch, 2048)
        
        # Step 7: Classification
        if hasattr(self, 'use_arcface') and self.use_arcface and labels is not None:
            logits = self.classifier(bottleneck, labels)
        else:
            logits = self.classifier(bottleneck)
        
        # Step 8: Decode to original modality dimensions
        vision_recon = self.vision_decoder(vision_universal_recon)      # 2048 → 1408
        language_recon = self.language_decoder(language_universal_recon) # 2048 → 7168
        
        # Step 9: Get projections for contrastive learning (from reconstructed universals)
        vision_proj, vision_temp = self.vision_projection(vision_universal_recon)
        language_proj, language_temp = self.language_projection(language_universal_recon)
        
        # Average temperatures
        temperature = (vision_temp + language_temp) / 2
        
        outputs = {
            'logits': logits,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'bottleneck': bottleneck,
            'vision_universal': vision_universal,  # Original, unmasked
            'language_universal': language_universal,  # Original, unmasked
            'vision_universal_recon': vision_universal_recon,
            'language_universal_recon': language_universal_recon,
            'vision_proj': vision_proj,  # For contrastive loss
            'language_proj': language_proj,  # For contrastive loss
            'temperature': temperature,
            'original_vision': original_vision,
            'original_language': original_language,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }
        
        return outputs


class HierarchicalUNet(nn.Module):
    """
    1D U-Net that processes universal embeddings
    Input shape: (batch, n_modalities, embedding_dim)
    Treats embedding_dim as the spatial dimension
    """
    
    def __init__(self, in_channels, out_channels, features, embedding_dim, use_attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        
        # Initial projection
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=3, padding=1)
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i in range(len(features)):
            in_feat = features[i-1] if i > 0 else features[0]
            out_feat = features[i]
            
            self.encoders.append(self._make_encoder_block(in_feat, out_feat))
            if i < len(features) - 1:  # No pooling after last encoder
                self.pools.append(nn.MaxPool1d(2))
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(features[-1], features[-1] * 2)
        
        if use_attention:
            self.attention = CrossModalAttention1D(features[-1] * 2)
        else:
            self.attention = None
        
        # Decoder path
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Build decoder in reverse order
        decoder_features = list(reversed(features))
        
        for i in range(len(decoder_features)):
            if i == 0:
                # First decoder: from bottleneck
                in_feat = features[-1] * 2
                out_feat = decoder_features[i]
            else:
                # Subsequent decoders
                in_feat = decoder_features[i-1]
                out_feat = decoder_features[i]
            
            # Upconv reduces channels from in_feat to out_feat
            self.upconvs.append(nn.ConvTranspose1d(in_feat, out_feat, kernel_size=2, stride=2))
            # Decoder takes concatenated features (out_feat from upconv + out_feat from skip)
            self.decoders.append(self._make_decoder_block(out_feat * 2, out_feat))
        
        # Final projection
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        """
        Forward pass through U-Net
        Input: (batch, n_modalities, embedding_dim)
        Output: (batch, n_modalities, embedding_dim)
        """
        # Initial projection
        x = self.init_conv(x)
        
        # Encoder path with skip connections
        encoder_features = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_features.append(x)
            if i < len(self.encoders) - 1:  # Don't pool after last encoder
                x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Apply attention if enabled
        if self.attention is not None:
            x = self.attention(x)
        
        # Decoder path
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            
            # Get skip connection (reverse order)
            skip_idx = len(encoder_features) - i - 1
            skip_features = encoder_features[skip_idx]
            
            # Ensure sizes match (handle odd dimensions)
            if x.shape[2] != skip_features.shape[2]:
                x = F.interpolate(x, size=skip_features.shape[2], mode='linear', align_corners=False)
            
            x = torch.cat([x, skip_features], dim=1)
            x = decoder(x)
        
        # Final projection back to n_modalities
        x = self.final_conv(x)
        
        # Ensure output has same embedding dimension as input
        if x.shape[2] != self.embedding_dim:
            x = F.interpolate(x, size=self.embedding_dim, mode='linear', align_corners=False)
        
        return x


class CrossModalAttention1D(nn.Module):
    """Cross-modal attention for 1D U-Net bottleneck"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.query = nn.Conv1d(channels, channels // 4, 1)
        self.key = nn.Conv1d(channels, channels // 4, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        
        self.output_conv = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(16, channels)
    
    def forward(self, x):
        B, C, L = x.shape
        
        # Self-attention across the sequence
        q = self.query(x).transpose(1, 2)  # (B, L, C/4)
        k = self.key(x)                     # (B, C/4, L)
        v = self.value(x).transpose(1, 2)   # (B, L, C)
        
        attention = F.softmax(torch.bmm(q, k) / np.sqrt(C // 4), dim=-1)
        out = torch.bmm(attention, v).transpose(1, 2)  # (B, C, L)
        
        out = self.output_conv(out)
        out = self.norm(out)
        
        return out + x  # Residual connection


# Keep all your existing training functions exactly as they are
def collate_fn(batch):
    """Custom collate function to handle batching"""
    vision = torch.stack([item['vision_embedding'] for item in batch])
    language = torch.stack([item['language_embedding'] for item in batch])
    labels = torch.stack([item['species_label'] for item in batch])
    
    return {
        'vision_embedding': vision,
        'language_embedding': language,
        'species_label': labels
    }


def compute_species_aware_contrastive_loss(vision_universal, language_universal, labels, 
                                          temperature=0.07, instance_weight=0.1):
    """
    Symmetric contrastive loss that treats all same-species pairs as positives.
    This handles the many-to-one nature of the data.
    
    Args:
        vision_universal: Vision embeddings
        language_universal: Language embeddings
        labels: Species labels
        temperature: Temperature for scaling
        instance_weight: Weight for instance-level alignment loss
    """
    batch_size = vision_universal.shape[0]
    device = vision_universal.device
    
    # Normalize embeddings
    vision_norm = F.normalize(vision_universal, p=2, dim=1)
    language_norm = F.normalize(language_universal, p=2, dim=1)
    
    # Compute all pairwise similarities
    similarities = torch.matmul(vision_norm, language_norm.T) / temperature
    
    # Create label matrix - True where labels match
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Vision → Language loss (all same-species are positives)
    exp_sim = torch.exp(similarities)
    # Sum over all positives (same species)
    positive_sum = (exp_sim * labels_matrix.float()).sum(dim=1)
    # Sum over all samples
    total_sum = exp_sim.sum(dim=1)
    loss_v2l = -torch.log(positive_sum / total_sum).mean()
    
    # Language → Vision loss (symmetric)
    exp_sim_T = torch.exp(similarities.T)
    positive_sum_T = (exp_sim_T * labels_matrix.float()).sum(dim=1)
    total_sum_T = exp_sim_T.sum(dim=1)
    loss_l2v = -torch.log(positive_sum_T / total_sum_T).mean()
    
    # Instance-level alignment loss (encourage diagonal to be high)
    # This helps maintain instance-specific information
    instance_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    instance_sims = similarities[instance_mask]
    instance_targets = torch.ones_like(instance_sims)
    instance_loss = F.mse_loss(instance_sims, instance_targets) * instance_weight
    
    # Combined loss
    loss = 0.5 * (loss_v2l + loss_l2v) + instance_loss
    
    return loss


def compute_species_aware_contrastive_loss_with_hard_negatives(vision_universal, language_universal, labels, 
                                                               temperature=0.07, hard_neg_ratio=0.5):
    """
    Enhanced contrastive loss with hard negative mining for better L->V retrieval
    """
    batch_size = vision_universal.shape[0]
    device = vision_universal.device
    
    # Normalize embeddings
    vision_norm = F.normalize(vision_universal, p=2, dim=1)
    language_norm = F.normalize(language_universal, p=2, dim=1)
    
    # L2 regularization on language embeddings to encourage spread
    language_l2_reg = torch.norm(language_universal, p=2, dim=1).mean() * 0.01
    
    # Compute all pairwise similarities
    similarities = torch.matmul(vision_norm, language_norm.T) / temperature
    
    # Create label matrix - True where labels match
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # For hard negative mining
    neg_mask = ~labels_matrix
    
    # Vision → Language loss with hard negatives
    with torch.no_grad():
        # Find hardest negatives for each vision sample
        neg_similarities_v2l = similarities * neg_mask.float() - 1000 * labels_matrix.float()
        k = max(1, int(batch_size * hard_neg_ratio))
        _, hard_indices_v2l = neg_similarities_v2l.topk(k, dim=1)
        
        # Create mask for hard negatives
        hard_neg_mask_v2l = torch.zeros_like(similarities).bool()
        hard_neg_mask_v2l.scatter_(1, hard_indices_v2l, True)
        hard_neg_mask_v2l = hard_neg_mask_v2l & neg_mask
    
    exp_sim = torch.exp(similarities)
    positive_sum = (exp_sim * labels_matrix.float()).sum(dim=1)
    # Include all positives and hard negatives
    hard_neg_sum = (exp_sim * hard_neg_mask_v2l.float()).sum(dim=1)
    loss_v2l = -torch.log(positive_sum / (positive_sum + hard_neg_sum + 1e-8)).mean()
    
    # Language → Vision loss with hard negatives (more important)
    with torch.no_grad():
        # Find hardest negatives for each language sample
        neg_similarities_l2v = similarities.T * neg_mask.float() - 1000 * labels_matrix.float()
        _, hard_indices_l2v = neg_similarities_l2v.topk(k, dim=1)
        
        hard_neg_mask_l2v = torch.zeros_like(similarities.T).bool()
        hard_neg_mask_l2v.scatter_(1, hard_indices_l2v, True)
        hard_neg_mask_l2v = hard_neg_mask_l2v & neg_mask
    
    exp_sim_T = torch.exp(similarities.T)
    positive_sum_T = (exp_sim_T * labels_matrix.float()).sum(dim=1)
    hard_neg_sum_T = (exp_sim_T * hard_neg_mask_l2v.float()).sum(dim=1)
    loss_l2v = -torch.log(positive_sum_T / (positive_sum_T + hard_neg_sum_T + 1e-8)).mean()
    
    # Weight L->V more heavily since it needs improvement
    loss = 0.3 * loss_v2l + 0.7 * loss_l2v + language_l2_reg
    
    return loss


# Keep all your existing train_epoch_fixed and evaluate_with_species_aware_retrieval functions
def train_epoch_fixed(model, loader, optimizer, device, lambda_rec=0.1, lambda_contrast=0.1,
                mask_config=None, temperature=0.07, epoch=0, use_hard_negatives=True,
                total_epochs=60, label_smoothing=0.0, lambda_center=1e-4, lambda_var=1e-3,
                instance_weight=0.1, language_noise=0.01, gradient_accumulation=1,
                monitor_collapse=False, contrast_ramp_epochs=0, contrast_ramp_start=0.1, 
                contrast_ramp_end=0.7):
    """Enhanced training with all the improvements"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    total_contrast_loss = 0
    correct = 0
    total = 0
    
    # Temperature scheduling with cosine warmup
    temperature = temperature_schedule(epoch, total_epochs, initial_temp=0.2, final_temp=temperature)
    
    # NEW: Contrast weight ramping
    if contrast_ramp_epochs > 0 and epoch < contrast_ramp_epochs:
        lambda_contrast = np.interp(epoch, [0, contrast_ramp_epochs-1], 
                                  [contrast_ramp_start, contrast_ramp_end])
        logger.info(f"Ramping contrast weight: epoch {epoch} -> λ_contrast = {lambda_contrast:.3f}")
    
    # For k-NN evaluation
    all_bottlenecks = []
    all_labels = []
    
    # NEW: Monitoring metrics
    paired_cosines = []
    vision_stds = []
    language_stds = []
    modal_distances = []
    
    # Initialize optimizer state for gradient accumulation
    optimizer.zero_grad()
    
    for i, batch in enumerate(loader):
        # Move to device
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        # Configure masking based on strategy
        mask_vision = False
        mask_language = False
        
        if mask_config:
            strategy = mask_config.get('strategy', 'none')
            if strategy == 'vision_only':
                mask_vision = True
            elif strategy == 'language_only':
                mask_language = True
            elif strategy == 'both':
                mask_vision = True
                mask_language = True
            elif strategy == 'alternate':
                if i % 2 == 0:
                    mask_language = True
                else:
                    mask_vision = True
            elif strategy == 'sample_level':
                # Mask entire samples instead of features
                mask_language = True
        
        # Add language noise before encoding if specified
        if language_noise > 0 and model.training and epoch > 5:
            language = language + language_noise * torch.randn_like(language)
        
        # Forward pass
        if hasattr(model, 'unet'):  # Check if it's the U-Net model
            outputs = model(
                vision, language, 
                mask_vision=mask_vision,
                mask_language=mask_language,
                vision_mask_ratio=mask_config.get('vision_ratio', 0.5),
                language_mask_ratio=mask_config.get('language_ratio', 0.5),
                mask_type=mask_config.get('mask_type', 'feature'),
                labels=labels  # Pass labels for ArcFace
            )
        else:
            # Standard model doesn't take labels parameter
            outputs = model(
                vision, language, 
                mask_vision=mask_vision,
                mask_language=mask_language,
                vision_mask_ratio=mask_config.get('vision_ratio', 0.5),
                language_mask_ratio=mask_config.get('language_ratio', 0.5),
                mask_type=mask_config.get('mask_type', 'feature')
            )
        
        # NEW: Monitor paired cosine similarity and collapse
        with torch.no_grad():
            v_norm = F.normalize(outputs['vision_universal'], p=2, dim=1)
            l_norm = F.normalize(outputs['language_universal'], p=2, dim=1)
            paired_cos = (v_norm * l_norm).sum(dim=1).mean()
            paired_cosines.append(paired_cos.item())
            
            if monitor_collapse:
                # Check if embeddings are collapsing
                vision_std = outputs['vision_universal'].std(dim=0).mean()
                language_std = outputs['language_universal'].std(dim=0).mean()
                vision_stds.append(vision_std.item())
                language_stds.append(language_std.item())
                
                # Check modality gap
                modal_distance = (outputs['vision_universal'].mean(0) - 
                                outputs['language_universal'].mean(0)).norm()
                modal_distances.append(modal_distance.item())
        
        # Classification loss with label smoothing
        if hasattr(model, 'use_arcface') and model.use_arcface:
            # ArcFace already computes the correct logits with margin
            cls_loss = F.cross_entropy(outputs['logits'], labels)
        elif label_smoothing > 0:
            # Create smoothed targets
            smooth_targets = torch.zeros_like(outputs['logits'])
            smooth_targets.fill_(label_smoothing / (outputs['logits'].size(1) - 1))
            smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - label_smoothing)
            cls_loss = F.kl_div(F.log_softmax(outputs['logits'], dim=1), smooth_targets, reduction='batchmean')
        else:
            cls_loss = F.cross_entropy(outputs['logits'], labels)
        
        # Reconstruction losses
        vision_rec_loss = 0
        language_rec_loss = 0
        
        # For the new U-Net architecture, we have two types of reconstruction:
        # 1. Universal-level reconstruction (where masking happens)
        # 2. Modality-level reconstruction (final output)
        
        if 'vision_universal_recon' in outputs and outputs.get('vision_universal_recon') is not None:
            # Universal-level reconstruction losses (where masking actually happens)
            if mask_vision and outputs['vision_mask'] is not None:
                mask = outputs['vision_mask']
                if mask_config.get('mask_type', 'feature') == 'token':
                    # Token-level masking
                    vision_universal_loss = F.mse_loss(
                        outputs['vision_universal_recon'], 
                        outputs['vision_universal']
                    )
                else:
                    # Feature-level masking in universal space
                    masked_errors = ((outputs['vision_universal_recon'] - outputs['vision_universal']) ** 2) * mask.float()
                    vision_universal_loss = masked_errors.sum() / (mask.float().sum() + 1e-6)
            else:
                vision_universal_loss = F.mse_loss(outputs['vision_universal_recon'], outputs['vision_universal'])
            
            if mask_language and outputs['language_mask'] is not None:
                mask = outputs['language_mask']
                if mask_config.get('mask_type', 'feature') == 'token':
                    language_universal_loss = F.mse_loss(
                        outputs['language_universal_recon'],
                        outputs['language_universal']
                    )
                else:
                    masked_errors = ((outputs['language_universal_recon'] - outputs['language_universal']) ** 2) * mask.float()
                    language_universal_loss = masked_errors.sum() / (mask.float().sum() + 1e-6)
            else:
                language_universal_loss = F.mse_loss(outputs['language_universal_recon'], outputs['language_universal'])
            
            # Combine universal and modality-level losses
            vision_modality_loss = F.mse_loss(outputs['vision_recon'], outputs['original_vision'])
            language_modality_loss = F.mse_loss(outputs['language_recon'], outputs['original_language'])
            
            # Weight universal reconstruction more heavily since that's where masking happens
            vision_rec_loss = 0.7 * vision_universal_loss + 0.3 * vision_modality_loss
            language_rec_loss = 0.7 * language_universal_loss + 0.3 * language_modality_loss
            
        else:
            # Fallback for standard architecture
            vision_rec_loss = F.mse_loss(outputs['vision_recon'], outputs['original_vision'])
            language_rec_loss = F.mse_loss(outputs['language_recon'], outputs['original_language'])
        
        # Normalize by dimension
        vision_rec_loss = vision_rec_loss / vision.shape[1]
        language_rec_loss = language_rec_loss / language.shape[1]
        
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        
        # Species-aware contrastive loss with learned temperature
        # Always use projected features from the model
        vision_features = outputs.get('vision_proj', outputs['vision_universal_recon'] if 'vision_universal_recon' in outputs else outputs['vision_universal'])
        language_features = outputs.get('language_proj', outputs['language_universal_recon'] if 'language_universal_recon' in outputs else outputs['language_universal'])
        learned_temp = outputs.get('temperature', temperature)
        
        if use_hard_negatives and epoch > 5:  # Start hard negatives after warmup
            contrast_loss = compute_species_aware_contrastive_loss_with_hard_negatives(
                vision_features,
                language_features,
                labels,
                temperature=learned_temp,
                hard_neg_ratio=0.5
            )
        else:
            contrast_loss = compute_species_aware_contrastive_loss(
                vision_features,
                language_features,
                labels,
                temperature=learned_temp,
                instance_weight=instance_weight
            )
        
        # Regularization penalties
        center_penalty = compute_center_penalty(outputs['vision_universal'], outputs['language_universal'])
        var_penalty = (compute_variance_penalty(outputs['vision_universal']) + 
                      compute_variance_penalty(outputs['language_universal'])) / 2
        
        # Center regularization to prevent drift
        center_loss = F.mse_loss(
            outputs['vision_universal'].mean(0), 
            outputs['language_universal'].mean(0)
        )
        
        # Total loss with dynamic reconstruction weight
        if epoch > 5 and total > 0 and correct / total > 0.99:
            # Drop reconstruction after warm-up if classification is perfect
            effective_lambda_rec = 0.0
        else:
            effective_lambda_rec = lambda_rec
            
        # Total loss with all components
        loss = (cls_loss + 
                effective_lambda_rec * rec_loss + 
                lambda_contrast * contrast_loss + 
                lambda_center * center_penalty +
                lambda_var * var_penalty)
        
        # Backward
        loss = loss / gradient_accumulation  # Scale loss for gradient accumulation
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % gradient_accumulation == 0 or (i + 1) == len(loader):
            # NEW: Monitor gradient norms
            if monitor_collapse and i % 10 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        total_contrast_loss += contrast_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store embeddings for k-NN
        all_bottlenecks.append(outputs['bottleneck'].detach().cpu())
        all_labels.append(labels.cpu())
        
        # Progress
        if i % 10 == 0:
            progress_str = f"\r  Batch {i}/{len(loader)}: Loss={loss.item():.4f}, " \
                          f"Acc={correct/total:.2%}, Contrast={contrast_loss.item():.3f}"
            if monitor_collapse:
                progress_str += f", PairedCos={paired_cos:.3f}"
                if 'total_norm' in locals():
                    progress_str += f", GradNorm={total_norm:.2f}"
            print(progress_str, end='', flush=True)
    
    # Compute k-NN accuracy
    all_bottlenecks = torch.cat(all_bottlenecks, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(all_bottlenecks)
    
    _, indices = knn.kneighbors(all_bottlenecks)
    knn_correct = 0
    for i, neighbors in enumerate(indices):
        neighbor_labels = all_labels[neighbors[1:]]  # Skip self
        if np.any(neighbor_labels == all_labels[i]):
            knn_correct += 1
    knn_acc = knn_correct / len(all_labels)
    
    # NEW: Print monitoring summary
    if monitor_collapse and paired_cosines:
        print(f"\n  Monitoring - PairedCos: {np.mean(paired_cosines):.3f}, "
              f"VisionStd: {np.mean(vision_stds):.3f}, "
              f"LangStd: {np.mean(language_stds):.3f}, "
              f"ModalDist: {np.mean(modal_distances):.3f}")
    
    print()  # New line
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, knn_acc, total_contrast_loss / total)


@torch.no_grad()
def evaluate_with_species_aware_retrieval(model, loader, device, lambda_rec=0.1):
    """
    Evaluation that properly accounts for many-to-one nature of vision-to-language mapping.
    Multiple images can have the same species (same language description).
    
    Key changes:
    - Retrieval is considered successful if ANY retrieved embedding has the correct species
    - Added Mean Reciprocal Rank (MRR) metric
    - Better instance alignment metrics
    """
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_rec_loss = 0
    correct = 0
    total = 0
    
    # For retrieval metrics
    all_vision_universal = []
    all_language_universal = []
    all_labels = []
    all_bottlenecks = []
    
    for batch in loader:
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        labels = batch['species_label'].to(device)
        
        # Pass labels to model for ArcFace compatibility
        if hasattr(model, 'use_arcface') and model.use_arcface:
            outputs = model(vision, language, labels=labels)
        else:
            outputs = model(vision, language)
        
        # Losses
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        vision_rec_loss = F.mse_loss(outputs['vision_recon'], vision) / vision.shape[1]
        language_rec_loss = F.mse_loss(outputs['language_recon'], language) / language.shape[1]
        rec_loss = (vision_rec_loss + language_rec_loss) / 2
        loss = cls_loss + lambda_rec * rec_loss
        
        total_loss += loss.item() * len(labels)
        total_cls_loss += cls_loss.item() * len(labels)
        total_rec_loss += rec_loss.item() * len(labels)
        pred = outputs['logits'].argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
        
        # Store embeddings
        all_vision_universal.append(outputs['vision_universal'])
        all_language_universal.append(outputs['language_universal'])
        all_labels.append(labels)
        all_bottlenecks.append(outputs['bottleneck'])
    
    # Compute retrieval metrics
    if len(all_vision_universal) > 0:
        all_vision = torch.cat(all_vision_universal, dim=0)
        all_language = torch.cat(all_language_universal, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_bottlenecks = torch.cat(all_bottlenecks, dim=0)
        
        # Normalize for cosine similarity
        vision_norm = F.normalize(all_vision, p=2, dim=1)
        language_norm = F.normalize(all_language, p=2, dim=1)
        bottleneck_norm = F.normalize(all_bottlenecks, p=2, dim=1)
        
        # Compute similarity matrix
        v2l_similarities = torch.matmul(vision_norm, language_norm.T)
        
        # 1. Vision → Language R@1 (species-aware)
        v2l_r1_correct = 0
        v2l_mrr = 0  # Mean Reciprocal Rank
        
        for i in range(len(all_labels)):
            similarities = v2l_similarities[i]
            
            # Get top-1 retrieval
            retrieved_idx = similarities.argmax().item()
            if all_labels[retrieved_idx] == all_labels[i]:
                v2l_r1_correct += 1
            
            # Calculate MRR - find rank of first correct species
            sorted_indices = similarities.argsort(descending=True)
            sorted_labels = all_labels[sorted_indices]
            correct_positions = (sorted_labels == all_labels[i]).nonzero(as_tuple=True)[0]
            if len(correct_positions) > 0:
                rank = correct_positions[0].item() + 1
                v2l_mrr += 1.0 / rank
                
        v2l_r1 = v2l_r1_correct / len(all_labels)
        v2l_mrr = v2l_mrr / len(all_labels)
        
        # 2. Vision → Language R@5 (species-aware)
        v2l_species_correct = 0
        for i in range(len(all_labels)):
            similarities = v2l_similarities[i]
            
            # Get top-5 retrievals
            top_k = min(5, len(similarities))
            _, top_indices = similarities.topk(top_k)
            
            # Check if any of the top-k have the same species
            retrieved_labels = all_labels[top_indices]
            if (retrieved_labels == all_labels[i]).any():
                v2l_species_correct += 1
        
        v2l_species_acc = v2l_species_correct / len(all_labels)
        
        # 3. Language → Vision R@1 and R@5 (species-aware)
        l2v_r1_correct = 0
        l2v_species_correct = 0
        l2v_mrr = 0
        l2v_similarities = v2l_similarities.T  # Transpose for L→V
        
        for i in range(len(all_labels)):
            similarities = l2v_similarities[i]
            
            # R@1
            retrieved_idx = similarities.argmax().item()
            if all_labels[retrieved_idx] == all_labels[i]:
                l2v_r1_correct += 1
            
            # R@5
            top_k = min(5, len(similarities))
            _, top_indices = similarities.topk(top_k)
            retrieved_labels = all_labels[top_indices]
            if (retrieved_labels == all_labels[i]).any():
                l2v_species_correct += 1
            
            # MRR
            sorted_indices = similarities.argsort(descending=True)
            sorted_labels = all_labels[sorted_indices]
            correct_positions = (sorted_labels == all_labels[i]).nonzero(as_tuple=True)[0]
            if len(correct_positions) > 0:
                rank = correct_positions[0].item() + 1
                l2v_mrr += 1.0 / rank
        
        l2v_r1 = l2v_r1_correct / len(all_labels)
        l2v_r5 = l2v_species_correct / len(all_labels)
        l2v_mrr = l2v_mrr / len(all_labels)
        
        # 4. Instance-level alignment (how well paired embeddings align)
        # This is important even with many-to-one mapping
        instance_similarities = torch.diag(v2l_similarities)
        instance_alignment = instance_similarities.mean().item()
        
        # Also compute the rank of the paired instance among same-species instances
        instance_ranks = []
        for i in range(len(all_labels)):
            similarities = v2l_similarities[i]
            same_species_mask = all_labels == all_labels[i]
            same_species_sims = similarities[same_species_mask]
            
            # Rank of the actual paired instance among same-species instances
            paired_sim = similarities[i]
            rank = (same_species_sims > paired_sim).sum().item() + 1
            instance_ranks.append(rank)
        
        avg_instance_rank = np.mean(instance_ranks)
        
        # 5. Embedding space clustering quality
        similarities = torch.matmul(bottleneck_norm, bottleneck_norm.T)
        
        # Mask out self-similarities
        mask = torch.eye(len(all_labels), device=device).bool()
        similarities.masked_fill_(mask, -float('inf'))
        
        # Find nearest neighbor
        _, indices = similarities.max(dim=1)
        
        # Check if nearest neighbor has same species
        nn_same_species = (all_labels[indices] == all_labels).float().mean().item()
        
        # Print detailed metrics
        logger.info(f"\nDetailed Retrieval Metrics:")
        logger.info(f"  V→L MRR: {v2l_mrr:.3f}")
        logger.info(f"  L→V MRR: {l2v_mrr:.3f}")
        logger.info(f"  Avg instance rank within species: {avg_instance_rank:.2f}")
        
    else:
        v2l_r1 = 0.0
        v2l_species_acc = 0.0
        l2v_r1 = 0.0
        instance_alignment = 0.0
        nn_same_species = 0.0
        l2v_r5 = 0.0
    
    return (total_loss / total, correct / total, total_cls_loss / total, 
            total_rec_loss / total, v2l_r1, v2l_species_acc, l2v_r1, l2v_r5,
            instance_alignment, nn_same_species)


# Keep your load_split function exactly as is
def load_split(config_path, max_species=None, train_ids_subset=None, test_ids_subset=None):
    """Load train/test split and species mapping, optionally filtered to most common species"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Use provided subsets or load all
    if train_ids_subset is None or test_ids_subset is None:
        train_ids = []
        test_ids = []
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                train_ids.append(obs_id)
            else:
                test_ids.append(obs_id)
        
        if train_ids_subset is not None:
            train_ids = train_ids_subset
        if test_ids_subset is not None:
            test_ids = test_ids_subset
    else:
        train_ids = train_ids_subset
        test_ids = test_ids_subset
    
    # Count species frequencies in the subset
    from collections import Counter
    species_counts = Counter()
    
    for obs_id in train_ids:
        if obs_id in config['observation_mappings']:
            species_counts[config['observation_mappings'][obs_id]['taxon_name']] += 1
    
    # Filter to top N species if requested
    if max_species and max_species < len(species_counts):
        top_species = set(species for species, _ in species_counts.most_common(max_species))
        logger.info(f"Filtering to top {max_species} most common species from {len(train_ids)} samples")
        
        # Show the selected species
        logger.info("Selected species (by frequency):")
        for i, (species, count) in enumerate(species_counts.most_common(max_species)):
            logger.info(f"  {i+1:2d}. {count:4d} samples: {species}")
    else:
        top_species = set(species_counts.keys())
    
    # Now filter observations to only include selected species
    filtered_train_ids = []
    filtered_test_ids = []
    
    for obs_id in train_ids:
        if obs_id in config['observation_mappings']:
            if config['observation_mappings'][obs_id]['taxon_name'] in top_species:
                filtered_train_ids.append(obs_id)
    
    for obs_id in test_ids:
        if obs_id in config['observation_mappings']:
            if config['observation_mappings'][obs_id]['taxon_name'] in top_species:
                filtered_test_ids.append(obs_id)
    
    # Create species mapping only for selected species
    species_to_idx = {species: idx for idx, species in enumerate(sorted(top_species))}
    logger.info(f"Filtered dataset: {len(filtered_train_ids)} train, {len(filtered_test_ids)} test, {len(species_to_idx)} species")
    
    return filtered_train_ids, filtered_test_ids, species_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--config', type=str, default='config/central_florida_split.json')
    parser.add_argument('--lambda-rec', type=float, default=0.1, help='Reconstruction loss weight')
    parser.add_argument('--lambda-contrast', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension for decoders')
    parser.add_argument('--universal-dim', type=int, default=2048, help='Universal embedding dimension')
    parser.add_argument('--subset', type=int, default=None, help='Use subset of data for testing')
    parser.add_argument('--mask-strategy', type=str, default='language_only',
                       choices=['none', 'vision_only', 'language_only', 'both', 'alternate'],
                       help='Masking strategy for training')
    parser.add_argument('--vision-mask-ratio', type=float, default=0.5)
    parser.add_argument('--language-mask-ratio', type=float, default=0.5)
    parser.add_argument('--max-species', type=int, default=None,
                       help='Maximum number of species to use (filters to most common)')
    parser.add_argument('--auto-species', action='store_true',
                       help='Automatically choose number of species to reach subset target')
    parser.add_argument('--use-all-data', action='store_true',
                       help='Use all data for training (no train/test split)')
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced sampling (equal samples per species)')
    parser.add_argument('--mask-type', type=str, default='feature', choices=['feature', 'sample'],
                       help='Masking type: feature (mask dimensions) or sample (mask entire embeddings)')
    parser.add_argument('--temperature', type=float, default=0.07, 
                       help='Temperature for contrastive loss')
    parser.add_argument('--save-dir', type=str, default='checkpoints_autoencoder',
                       help='Directory to save checkpoints')
    
    # New U-Net specific arguments
    parser.add_argument('--use-unet', action='store_true', 
                       help='Use hierarchical U-Net instead of simple fusion')
    parser.add_argument('--spatial-size', type=int, default=32,
                       help='Spatial size for U-Net processing')
    parser.add_argument('--unet-features', type=int, nargs='+', default=[32, 64, 128, 256],
                       help='Feature dimensions for U-Net levels')
    parser.add_argument('--use-spatial-attention', action='store_true',
                       help='Use cross-modal spatial attention in U-Net')
    
    parser.add_argument('--use-hard-negatives', action='store_true',
                       help='Use hard negative mining for contrastive loss')
    parser.add_argument('--hard-neg-ratio', type=float, default=0.5,
                       help='Ratio of hard negatives to mine')
    parser.add_argument('--add-vision-noise', action='store_true',
                       help='Add noise to vision embeddings for robustness')
    parser.add_argument('--l2v-weight', type=float, default=0.7,
                       help='Weight for L->V loss (vs V->L)')
    parser.add_argument('--use-arcface', action='store_true',
                       help='Use ArcFace loss for classification')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing for classification')
    parser.add_argument('--lambda-center', type=float, default=1e-4,
                       help='Weight for center alignment penalty')
    parser.add_argument('--lambda-var', type=float, default=5e-4,
                       help='Weight for variance penalty')
    parser.add_argument('--use-balanced-sampler', action='store_true',
                       help='Use balanced species sampler')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--instance-loss-weight', type=float, default=0.1,
                       help='Weight for instance-level alignment in contrastive loss')
    parser.add_argument('--language-noise', type=float, default=0.01,
                       help='Noise level for language embeddings to create diversity')
    
    # NEW: Phased training arguments
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--freeze-classifier', action='store_true',
                       help='Freeze classifier weights during training')
    parser.add_argument('--train-only-projections', action='store_true',
                       help='Only train projection heads and encoders')
    parser.add_argument('--contrast-ramp-epochs', type=int, default=0,
                       help='Number of epochs to ramp contrast weight (0=no ramp)')
    parser.add_argument('--contrast-ramp-start', type=float, default=0.1,
                       help='Starting value for contrast weight ramp')
    parser.add_argument('--contrast-ramp-end', type=float, default=0.7,
                       help='Ending value for contrast weight ramp')
    parser.add_argument('--monitor-collapse', action='store_true',
                       help='Monitor for embedding collapse')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load config and handle balanced sampling
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # [Keep all your existing data loading logic exactly as is]
    if args.use_all_data:
        # Use all available data for training
        logger.info("Using all data for training (no train/test split)")
        
        from collections import Counter
        
        # Get ALL observation IDs
        all_obs_ids = list(config['observation_mappings'].keys())
        
        # Count species
        species_counts = Counter()
        obs_ids_by_species = {}
        
        for obs_id, meta in config['observation_mappings'].items():
            species = meta['taxon_name']
            species_counts[species] += 1
            if species not in obs_ids_by_species:
                obs_ids_by_species[species] = []
            obs_ids_by_species[species].append(obs_id)
        
        # Select top species
        if args.max_species:
            top_species = [s for s, _ in species_counts.most_common(args.max_species)]
            selected_obs_ids = []
            for species in top_species:
                selected_obs_ids.extend(obs_ids_by_species[species])
            
            species_mapping = {species: idx for idx, species in enumerate(sorted(top_species))}
            logger.info(f"Selected {len(top_species)} species with {len(selected_obs_ids)} total observations")
        else:
            selected_obs_ids = all_obs_ids
            all_species = sorted(species_counts.keys())
            species_mapping = {species: idx for idx, species in enumerate(all_species)}
        
        # Apply subset if requested
        if args.subset and args.subset < len(selected_obs_ids):
            selected_obs_ids = selected_obs_ids[:args.subset]
        
        # Use same data for both "train" and "test" 
        # (test is just for evaluation metrics, not real testing)
        train_ids = selected_obs_ids
        
        # Use more samples for stable evaluation metrics
        eval_size = min(1000, len(selected_obs_ids) // 5)  # 20% of data or 1000, whichever is smaller
        test_ids = selected_obs_ids[:eval_size]
        
        logger.info(f"Total samples: {len(train_ids)}")
        logger.info(f"Using {len(test_ids)} samples for evaluation metrics (contrastive signals need more data)")
        
    elif args.balanced and args.subset and args.max_species:
        # Smart balanced sampling: maximize samples while respecting data limits
        logger.info(f"Using smart balanced sampling: up to {args.subset} samples across {args.max_species} species")
        
        from collections import Counter
        
        # Count species in training
        species_counts = Counter()
        train_ids_by_species = {}
        
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                species = meta['taxon_name']
                species_counts[species] += 1
                if species not in train_ids_by_species:
                    train_ids_by_species[species] = []
                train_ids_by_species[species].append(obs_id)
        
        # Get top species by count
        top_species = species_counts.most_common(args.max_species)
        
        # Strategy: Take all available samples from top species, up to subset limit
        train_ids_subset = []
        test_ids_subset = []
        selected_species = []
        samples_per_species_actual = {}
        
        # First pass: see how many samples we can get from top species
        total_available = sum(min(count, args.subset // args.max_species * 2) for species, count in top_species)
        
        if total_available < args.subset:
            # Take all samples from top species
            for species, count in top_species:
                species_train_ids = train_ids_by_species[species]
                train_ids_subset.extend(species_train_ids)
                selected_species.append(species)
                samples_per_species_actual[species] = len(species_train_ids)
        else:
            # Distribute samples proportionally
            # Start with minimum samples per species
            min_samples = 20  # Minimum to ensure decent learning
            remaining_samples = args.subset
            
            # First, allocate minimum to each species
            for species, count in top_species:
                if count >= min_samples and remaining_samples >= min_samples:
                    take = min(min_samples, count)
                    species_train_ids = train_ids_by_species[species][:take]
                    train_ids_subset.extend(species_train_ids)
                    selected_species.append(species)
                    samples_per_species_actual[species] = take
                    remaining_samples -= take
            
            # Then distribute remaining samples proportionally
            if remaining_samples > 0 and selected_species:
                total_count = sum(count for species, count in top_species if species in selected_species)
                
                for species in selected_species:
                    if remaining_samples <= 0:
                        break
                    
                    species_count = species_counts[species]
                    current_samples = samples_per_species_actual[species]
                    
                    # Proportional allocation of remaining samples
                    proportion = species_count / total_count
                    additional = int(remaining_samples * proportion)
                    
                    # Don't exceed available samples for this species
                    can_add = min(additional, species_count - current_samples)
                    
                    if can_add > 0:
                        species_train_ids = train_ids_by_species[species][current_samples:current_samples + can_add]
                        train_ids_subset.extend(species_train_ids)
                        samples_per_species_actual[species] += len(species_train_ids)
                        remaining_samples -= len(species_train_ids)
        
        # Add test samples for selected species
        min_test_per_species = 5  # Ensure at least 5 test samples per species
        for species in selected_species:
            test_ids_for_species = [obs_id for obs_id, meta in config['observation_mappings'].items() 
                                  if meta['split'] == 'test' and meta['taxon_name'] == species]
            
            # If not enough test samples, take some from training
            if len(test_ids_for_species) < min_test_per_species:
                # Get species training IDs we haven't used yet
                species_all_train = train_ids_by_species[species]
                used_count = samples_per_species_actual[species]
                available_for_test = species_all_train[used_count:]
                
                # Move some training samples to test to ensure minimum
                needed = min_test_per_species - len(test_ids_for_species)
                if len(available_for_test) >= needed:
                    # Remove from train and add to test
                    move_to_test = available_for_test[:needed]
                    test_ids_for_species.extend(move_to_test)
                    
                    # Remove these from train_ids_subset
                    train_ids_subset = [id for id in train_ids_subset if id not in move_to_test]
                    
                    logger.info(f"  Moved {needed} samples from train to test for species {species}")
            
            test_ids_subset.extend(test_ids_for_species[:min_test_per_species])
        
        # Create species mapping
        species_mapping = {species: idx for idx, species in enumerate(sorted(selected_species))}
        
        # Log distribution
        logger.info(f"Smart balanced dataset created:")
        logger.info(f"  Species: {len(selected_species)}")
        logger.info(f"  Train samples: {len(train_ids_subset)}")
        logger.info(f"  Test samples: {len(test_ids_subset)}")
        logger.info(f"\nSamples per species:")
        for species in sorted(selected_species):
            logger.info(f"    {species}: {samples_per_species_actual[species]} samples")
        
        train_ids = train_ids_subset
        test_ids = test_ids_subset
        
    else:
        # Original loading logic
        all_train_ids = []
        all_test_ids = []
        for obs_id, meta in config['observation_mappings'].items():
            if meta['split'] == 'train':
                all_train_ids.append(obs_id)
            else:
                all_test_ids.append(obs_id)
        
        logger.info(f"Total available: {len(all_train_ids)} train, {len(all_test_ids)} test")
        
        # Apply subset FIRST
        if args.subset:
            train_ids_subset = all_train_ids[:args.subset]
            test_ids_subset = all_test_ids[:min(args.subset//5, len(all_test_ids))]
        else:
            train_ids_subset = all_train_ids
            test_ids_subset = all_test_ids
        
        # Then filter by species
        train_ids, test_ids, species_mapping = load_split(
            str(config_path), 
            max_species=args.max_species,
            train_ids_subset=train_ids_subset,
            test_ids_subset=test_ids_subset
        )
    
    logger.info(f"Split: {len(train_ids)} train, {len(test_ids)} test")
    
    # Change to dashboard directory
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Initialize cache
        cache = UnifiedDataCache("dataset_config.json")
        
        # Create datasets
        logger.info("Creating training dataset...")
        train_dataset = EfficientMultimodalDataset(
            train_ids, cache, species_mapping=species_mapping
        )
        
        logger.info("Creating test dataset...")
        test_dataset = EfficientMultimodalDataset(
            test_ids, cache, species_mapping=species_mapping
        )
        
        # Debug: Check test set distribution
        logger.info("Analyzing test set distribution...")
        test_labels = []
        for i in range(len(test_dataset)):
            test_labels.append(test_dataset.get_species_label(i))
        
        from collections import Counter
        test_distribution = Counter(test_labels)
        logger.info(f"Test set species distribution: {dict(test_distribution)}")
        logger.info(f"Species in test set: {len(test_distribution)} out of {len(species_mapping)}")
        
        # Create loaders
        logger.info("Creating data loaders...")
        if args.use_balanced_sampler:
            logger.info("Creating BalancedSpeciesSampler for training data...")
            train_sampler = BalancedSpeciesSampler(train_dataset)
            logger.info("BalancedSpeciesSampler created successfully")
            logger.info(f"Creating train DataLoader with batch_size={args.batch_size}, num_workers={args.num_workers}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=(device == 'cuda'),
                persistent_workers=(args.num_workers > 0)
            )
            logger.info("Train DataLoader created successfully")
        else:
            logger.info(f"Creating train DataLoader without sampler, batch_size={args.batch_size}, num_workers={args.num_workers}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=(device == 'cuda'),
                persistent_workers=(args.num_workers > 0)
            )
            logger.info("Train DataLoader created successfully")
        
        logger.info("Creating test DataLoader...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda'),
            persistent_workers=(args.num_workers > 0)
        )
        logger.info("Test DataLoader created successfully")
        
        # Create model - now with optional U-Net
        logger.info("Initializing model...")
        if args.use_unet:
            logger.info("Using Hierarchical U-Net architecture")
            logger.info(f"Creating U-Net model with features={args.unet_features}")
            model = MultimodalAutoencoderWithHierarchicalUNet(
                num_classes=train_dataset.num_classes,
                hidden_dim=args.hidden_dim,
                universal_dim=args.universal_dim,
                unet_features=args.unet_features,
                use_spatial_attention=args.use_spatial_attention
            )
            # Set attributes after model creation
            model.use_arcface = args.use_arcface
            model.add_embedding_noise = args.add_vision_noise
            if args.use_arcface:
                # Replace classifier with ArcFace
                model.classifier = ArcFaceLoss(args.universal_dim, train_dataset.num_classes, s=30.0, m=0.2)
            logger.info("Moving model to device...")
            model = model.to(device)
            logger.info("Model moved to device successfully")
        else:
            logger.info("Using standard MLP fusion architecture")
            # Use the MultimodalAutoencoder class defined in this file
            model = MultimodalAutoencoder(
                num_classes=train_dataset.num_classes,
                hidden_dim=args.hidden_dim,
                universal_dim=args.universal_dim
            )
            logger.info("Moving model to device...")
            model = model.to(device)
            logger.info("Model moved to device successfully")
        
        # NEW: Load from checkpoint if specified
        if args.resume_from:
            logger.info(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            
            # Handle loading from different model types
            if 'model_type' in checkpoint:
                saved_type = checkpoint['model_type']
                current_type = 'hierarchical_unet' if args.use_unet else 'standard_mlp'
                
                if saved_type != current_type:
                    logger.warning(f"Loading {saved_type} weights into {current_type} model")
                    logger.warning("Only compatible layers will be loaded")
            
            # Load state dict with strict=False to handle architecture mismatches
            incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            
            # Load species mapping if available
            if 'species_mapping' in checkpoint:
                loaded_species = checkpoint['species_mapping']
                if len(loaded_species) != len(species_mapping):
                    logger.warning(f"Species count mismatch: checkpoint has {len(loaded_species)}, "
                                 f"current has {len(species_mapping)}")
        
        # NEW: Freeze classifier if specified
        if args.freeze_classifier:
            logger.info("Freezing classifier weights")
            for p in model.classifier.parameters():
                p.requires_grad = False
        
        # NEW: Train only projections and encoders if specified
        if args.train_only_projections:
            logger.info("Training only projection heads and encoders")
            # First freeze everything
            for p in model.parameters():
                p.requires_grad = False
            
            # Then unfreeze projections and encoders
            for name, p in model.named_parameters():
                if "projection" in name or "encoder" in name:
                    p.requires_grad = True
                    logger.info(f"  Unfrozen: {name}")
        
        logger.info(f"Model initialized:")
        logger.info(f"  Vision: 1408 → {args.universal_dim}")
        logger.info(f"  Language: 7168 → {args.universal_dim}")
        if args.use_unet:
            logger.info(f"  U-Net processes universal embeddings: (batch, 2, {args.universal_dim})")
            logger.info(f"  U-Net features: {args.unet_features}")
            logger.info(f"  Cross-modal attention: {args.use_spatial_attention}")
        logger.info(f"  Decoders: 2-layer MLPs with hidden_dim={args.hidden_dim}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Training setup
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Use ReduceLROnPlateau instead of CosineAnnealingLR
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize score
            factor=0.5,  # Reduce LR by half
            patience=5   # Wait 5 epochs before reducing
        )
        
        # Masking configuration
        mask_config = {
            'strategy': args.mask_strategy,
            'vision_ratio': args.vision_mask_ratio,
            'language_ratio': args.language_mask_ratio,
            'mask_type': args.mask_type,
            'add_vision_noise': args.add_vision_noise
        }
        
        print("\n" + "="*80)
        print("🚀 STARTING MULTIMODAL AUTOENCODER TRAINING")
        print("="*80)
        print(f"Architecture: {'Hierarchical U-Net' if args.use_unet else 'Standard MLP'} Fusion")
        print(f"Vision(1408) + Language(7168) → Universal({args.universal_dim})")
        print(f"Decoders: 2-layer MLPs (hidden_dim={args.hidden_dim})")
        print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        print(f"Masking strategy: {args.mask_strategy}")
        if args.mask_strategy != 'none':
            print(f"  Vision mask ratio: {args.vision_mask_ratio}")
            print(f"  Language mask ratio: {args.language_mask_ratio}")
            print(f"  Mask type: {args.mask_type}")
        print("="*80 + "\n")
        
        # Track multiple metrics for best model
        best_metrics = {
            'epoch': 0,
            'test_acc': 0,
            'v2l_r1': 0,
            'v2l_r5': 0,
            'l2v_r1': 0,
            'l2v_r5': 0,
            'instance_alignment': 0,
            'combined_score': 0
        }
        
        # Track the best value for EACH metric individually
        best_individual_metrics = {
            'test_acc': {'value': 0, 'epoch': 0},
            'v2l_r1': {'value': 0, 'epoch': 0},
            'v2l_r5': {'value': 0, 'epoch': 0},
            'l2v_r1': {'value': 0, 'epoch': 0},
            'l2v_r5': {'value': 0, 'epoch': 0},
            'instance_alignment': {'value': 0, 'epoch': 0},
            'nn_same_species': {'value': 0, 'epoch': 0},
            'train_knn': {'value': 0, 'epoch': 0},
            'avg_contrast_loss': {'value': float('inf'), 'epoch': 0}  # Lower is better
        }
        
        # For tracking history
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'v2l_r1': [],
            'v2l_r5': [],
            'l2v_r1': [], 
            'l2v_r5': [],
            'alignment': [],
            'nn_same_species': [],
            'train_knn': [],
            'contrast_loss': []
        }
        
        for epoch in range(args.epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
            
            # Train with species-aware contrastive loss
            print("Training...")
            train_loss, train_acc, train_cls, train_rec, train_knn, train_contrast = train_epoch_fixed(
                model, train_loader, optimizer, device, 
                lambda_rec=args.lambda_rec,
                lambda_contrast=args.lambda_contrast,
                mask_config=mask_config,
                temperature=args.temperature,
                epoch=epoch,
                use_hard_negatives=args.use_hard_negatives,
                total_epochs=args.epochs,
                label_smoothing=args.label_smoothing,
                lambda_center=args.lambda_center,
                lambda_var=args.lambda_var,
                instance_weight=args.instance_loss_weight,
                language_noise=args.language_noise,
                gradient_accumulation=args.gradient_accumulation,
                monitor_collapse=args.monitor_collapse,
                contrast_ramp_epochs=args.contrast_ramp_epochs,
                contrast_ramp_start=args.contrast_ramp_start,
                contrast_ramp_end=args.contrast_ramp_end
            )
            
            print(f"\n📊 Train Results:")
            print(f"   Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, Rec: {train_rec:.4f}, Contrast: {train_contrast:.4f})")
            print(f"   Classification Accuracy: {train_acc:.2%}")
            print(f"   k-NN Accuracy (k=5): {train_knn:.2%}")
            
            # Evaluate with species-aware metrics
            print("\nEvaluating...")
            test_loss, test_acc, test_cls, test_rec, v2l_r1, v2l_r5, l2v_r1, l2v_r5, instance_sim, nn_species = evaluate_with_species_aware_retrieval(
                model, test_loader, device, args.lambda_rec
            )
            
            print(f"\n📊 Test Results:")
            print(f"   Loss: {test_loss:.4f} (Cls: {test_cls:.4f}, Rec: {test_rec:.4f})")
            print(f"   Classification Accuracy: {test_acc:.2%}")
            print(f"   V→L Retrieval R@1: {v2l_r1:.2%}")
            print(f"   V→L Species R@5: {v2l_r5:.2%}")
            print(f"   L→V Species R@1: {l2v_r1:.2%}")
            print(f"   L→V Species R@5: {l2v_r5:.2%}")
            print(f"   Instance Alignment: {instance_sim:.3f}")
            print(f"   NN Same Species: {nn_species:.2%}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['v2l_r1'].append(v2l_r1)
            history['v2l_r5'].append(v2l_r5)
            history['l2v_r1'].append(l2v_r1)
            history['l2v_r5'].append(l2v_r5)
            history['alignment'].append(instance_sim)
            history['nn_same_species'].append(nn_species)
            history['train_knn'].append(train_knn)
            history['contrast_loss'].append(train_contrast)
            
            # Update best individual metrics
            current_metrics = {
                'test_acc': test_acc,
                'v2l_r1': v2l_r1,
                'v2l_r5': v2l_r5,
                'l2v_r1': l2v_r1,
                'l2v_r5': l2v_r5,
                'instance_alignment': instance_sim,
                'nn_same_species': nn_species,
                'train_knn': train_knn,
                'avg_contrast_loss': train_contrast
            }
            
            # Check if any individual metric is best
            new_bests = []
            for metric_name, metric_value in current_metrics.items():
                if metric_name == 'avg_contrast_loss':
                    # Lower is better for loss
                    if metric_value < best_individual_metrics[metric_name]['value']:
                        best_individual_metrics[metric_name]['value'] = metric_value
                        best_individual_metrics[metric_name]['epoch'] = epoch + 1
                        new_bests.append(f"{metric_name}: {metric_value:.4f}")
                else:
                    # Higher is better for accuracy/retrieval metrics
                    if metric_value > best_individual_metrics[metric_name]['value']:
                        best_individual_metrics[metric_name]['value'] = metric_value
                        best_individual_metrics[metric_name]['epoch'] = epoch + 1
                        new_bests.append(f"{metric_name}: {metric_value:.2%}" if metric_value <= 1 else f"{metric_name}: {metric_value:.3f}")
            
            if new_bests:
                print(f"\n🌟 New best(s): {', '.join(new_bests)}")
            
            # Create alignment visualization every 3 epochs
            if (epoch + 1) % 3 == 0 or epoch == 0:
                print("\n📊 Creating alignment visualization...")
                align_dir = Path(args.save_dir) / 'alignment_visualizations'
                align_dir.mkdir(exist_ok=True)
                
                avg_sim, v2l_acc = create_alignment_visualization(
                    model, test_loader, device, epoch + 1, align_dir
                )
            
            # Calculate combined score for determining best model
            combined_score = (test_acc + v2l_r1 + v2l_r5 + l2v_r1 + l2v_r5 + instance_sim) / 6
            
            # Update best metrics if this is better
            if combined_score > best_metrics['combined_score']:
                best_metrics.update({
                    'epoch': epoch + 1,
                    'test_acc': test_acc,
                    'v2l_r1': v2l_r1,
                    'v2l_r5': v2l_r5,
                    'l2v_r1': l2v_r1,
                    'l2v_r5': l2v_r5,
                    'instance_alignment': instance_sim,
                    'combined_score': combined_score
                })
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'v2l_r1': v2l_r1,
                    'v2l_species_r5': v2l_r5,
                    'l2v_species_r1': l2v_r1,
                    'l2v_species_r5': l2v_r5,
                    'instance_alignment': instance_sim,
                    'nn_same_species': nn_species,
                    'train_knn': train_knn,
                    'args': vars(args),
                    'species_mapping': species_mapping,
                    'history': history,
                    'best_individual_metrics': best_individual_metrics,
                    'model_type': 'hierarchical_unet' if args.use_unet else 'standard_mlp'
                }, Path(args.save_dir) / 'autoencoder_best.pth')
                print(f"\n🎯 NEW BEST MODEL! Combined Score: {combined_score:.3f}")
            else:
                print(f"\n   Best model: Score={best_metrics['combined_score']:.3f}, Acc={best_metrics['test_acc']:.2%} (Epoch {best_metrics['epoch']})")
            
            # Update scheduler with combined score
            scheduler.step(combined_score)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print(f"🏆 Best Overall Model (Epoch {best_metrics['epoch']}):")
        print(f"   Architecture: {'Hierarchical U-Net' if args.use_unet else 'Standard MLP'}")
        print(f"   Combined Score: {best_metrics['combined_score']:.3f}")
        print(f"   Test Accuracy: {best_metrics['test_acc']:.2%}")
        print(f"   V→L Retrieval: R@1={best_metrics['v2l_r1']:.2%}, R@5={best_metrics['v2l_r5']:.2%}")
        print(f"   L→V Retrieval: R@1={best_metrics['l2v_r1']:.2%}, R@5={best_metrics['l2v_r5']:.2%}")
        print(f"   Instance Alignment: {best_metrics['instance_alignment']:.3f}")
        
        print(f"\n📊 Best Individual Metrics Achieved:")
        for metric_name, metric_info in best_individual_metrics.items():
            if metric_name == 'avg_contrast_loss':
                print(f"   {metric_name}: {metric_info['value']:.4f} (Epoch {metric_info['epoch']})")
            elif metric_info['value'] <= 1:
                print(f"   {metric_name}: {metric_info['value']:.2%} (Epoch {metric_info['epoch']})")
            else:
                print(f"   {metric_name}: {metric_info['value']:.3f} (Epoch {metric_info['epoch']})")
        
        # Save final metrics summary
        metrics_summary = {
            'best_combined_model': best_metrics,
            'best_individual_metrics': best_individual_metrics,
            'final_history': {k: v[-1] if v else None for k, v in history.items()},
            'training_config': vars(args),
            'model_type': 'hierarchical_unet' if args.use_unet else 'standard_mlp'
        }
        
        with open(Path(args.save_dir) / 'training_metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"\n📁 Saved metrics summary to {Path(args.save_dir) / 'training_metrics_summary.json'}")
        print("="*80)
        
    finally:
        os.chdir(original_dir)


def transfer_weights_mlp_to_unet(mlp_checkpoint_path, save_path, device='cuda'):
    """Transfer weights from MLP model to U-Net model"""
    import torch
    
    # Load MLP checkpoint
    mlp_ckpt = torch.load(mlp_checkpoint_path, map_location=device)
    mlp_state = mlp_ckpt['model_state_dict']
    
    # Get configuration from checkpoint
    args = mlp_ckpt.get('args', {})
    num_classes = len(mlp_ckpt.get('species_mapping', {}))
    
    # Create U-Net model with same configuration
    unet_model = MultimodalAutoencoderWithHierarchicalUNet(
        num_classes=num_classes,
        universal_dim=args.get('universal_dim', 1024),
        hidden_dim=args.get('hidden_dim', 256),
        unet_features=[32, 64, 128, 256]
    )
    
    # Get U-Net state dict
    unet_state = unet_model.state_dict()
    
    # Transfer compatible weights
    transferred = {}
    skipped = []
    
    for name, param in mlp_state.items():
        if name in unet_state:
            if unet_state[name].shape == param.shape:
                transferred[name] = param
                logger.info(f"✓ Transferred: {name} {param.shape}")
            else:
                skipped.append((name, f"shape mismatch: {param.shape} vs {unet_state[name].shape}"))
        else:
            skipped.append((name, "not in U-Net model"))
    
    # Load transferred weights
    unet_model.load_state_dict(transferred, strict=False)
    
    # Save new checkpoint
    torch.save({
        'model_state_dict': unet_model.state_dict(),
        'transferred_from': mlp_checkpoint_path,
        'args': args,
        'species_mapping': mlp_ckpt.get('species_mapping', {}),
        'model_type': 'hierarchical_unet'
    }, save_path)
    
    logger.info(f"\nTransfer complete!")
    logger.info(f"  Transferred: {len(transferred)} layers")
    logger.info(f"  Skipped: {len(skipped)} layers")
    
    if skipped:
        logger.info("\nSkipped layers:")
        for name, reason in skipped[:10]:  # Show first 10
            logger.info(f"  - {name}: {reason}")
        if len(skipped) > 10:
            logger.info(f"  ... and {len(skipped) - 10} more")
    
    return save_path


def check_embedding_collapse(model, loader, device):
    """Check for signs of embedding collapse"""
    model.eval()
    
    vision_stds = []
    language_stds = []
    paired_cosines = []
    modal_distances = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:  # Sample 10 batches
                break
                
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            labels = batch['species_label'].to(device)
            
            outputs = model(vision, language)
            
            # Check embedding statistics
            vision_std = outputs['vision_universal'].std(dim=0).mean().item()
            language_std = outputs['language_universal'].std(dim=0).mean().item()
            
            # Check paired cosine similarity
            v_norm = F.normalize(outputs['vision_universal'], p=2, dim=1)
            l_norm = F.normalize(outputs['language_universal'], p=2, dim=1)
            paired_cos = (v_norm * l_norm).sum(dim=1).mean().item()
            
            # Check modality gap
            modal_dist = (outputs['vision_universal'].mean(0) - 
                         outputs['language_universal'].mean(0)).norm().item()
            
            vision_stds.append(vision_std)
            language_stds.append(language_std)
            paired_cosines.append(paired_cos)
            modal_distances.append(modal_dist)
    
    # Analyze results
    results = {
        'vision_std': np.mean(vision_stds),
        'language_std': np.mean(language_stds),
        'paired_cosine': np.mean(paired_cosines),
        'modal_distance': np.mean(modal_distances)
    }
    
    # Check for problems
    problems = []
    if results['vision_std'] < 0.1:
        problems.append("Vision embeddings may be collapsing (std < 0.1)")
    if results['language_std'] < 0.1:
        problems.append("Language embeddings may be collapsing (std < 0.1)")
    if results['paired_cosine'] < 0.2:
        problems.append("Poor vision-language alignment (cosine < 0.2)")
    if results['modal_distance'] > 10:
        problems.append("Large modality gap (distance > 10)")
    
    logger.info("\nEmbedding Health Check:")
    logger.info(f"  Vision STD: {results['vision_std']:.3f}")
    logger.info(f"  Language STD: {results['language_std']:.3f}")
    logger.info(f"  Paired Cosine: {results['paired_cosine']:.3f}")
    logger.info(f"  Modal Distance: {results['modal_distance']:.3f}")
    
    if problems:
        logger.warning("\n⚠️  Problems detected:")
        for problem in problems:
            logger.warning(f"  - {problem}")
    else:
        logger.info("\n✅ Embeddings look healthy!")
    
    return results


if __name__ == "__main__":
    # Check if this is a weight transfer request
    if len(sys.argv) > 1 and sys.argv[1] == 'transfer':
        # Usage: python unet_mlp.py transfer checkpoint.pth output.pth
        if len(sys.argv) != 4:
            print("Usage: python unet_mlp.py transfer <input_checkpoint> <output_checkpoint>")
            sys.exit(1)
        
        transfer_weights_mlp_to_unet(sys.argv[2], sys.argv[3])
    else:
        main()
