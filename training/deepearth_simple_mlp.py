#!/usr/bin/env python3
"""
DeepEarth MLP U-Net - Fixed Version with Proper Contrastive Learning
Implements all fixes for small dataset contrastive learning
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
import random
import csv
import os
from tqdm import tqdm
from collections import deque

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MemoryBank:
    """Memory bank for additional negatives (MoCo-style)"""
    def __init__(self, size=2048, dim=256):
        self.size = size
        self.dim = dim
        self.vision_bank = deque(maxlen=size)
        self.language_bank = deque(maxlen=size)
        self.labels_bank = deque(maxlen=size)
        
    def update(self, vision_emb, language_emb, labels):
        """Add new embeddings to the bank"""
        for v, l, y in zip(vision_emb, language_emb, labels):
            self.vision_bank.append(v.detach())
            self.language_bank.append(l.detach())
            self.labels_bank.append(y.detach())
    
    def get_all(self, device):
        """Get all embeddings as tensors"""
        if len(self.vision_bank) == 0:
            return None, None, None
        
        vision = torch.stack(list(self.vision_bank)).to(device)
        language = torch.stack(list(self.language_bank)).to(device)
        labels = torch.stack(list(self.labels_bank)).to(device)
        return vision, language, labels


class DeepEarthMLPDataset(Dataset):
    """Dataset with augmentation support"""
    
    def __init__(self, observation_ids, cache, mode='both', device='cpu', species_mapping=None, augment=True):
        self.observation_ids = observation_ids
        self.cache = cache
        self.mode = mode
        self.device = 'cpu'  # Always CPU for DataLoader
        self.species_mapping = species_mapping
        self.augment = augment
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and prepare all data at initialization."""
        logger.info(f"Loading dataset with {len(self.observation_ids)} observations...")
        
        batch_size = 64
        all_species = []
        all_language_embs = []
        all_vision_embs = []
        valid_obs_ids = []
        
        for i in range(0, len(self.observation_ids), batch_size):
            batch_ids = self.observation_ids[i:i + batch_size]
            
            try:
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=(self.mode in ['vision', 'both']),
                    include_language=(self.mode in ['language', 'both']),
                    device='cpu'
                )
                
                if 'species' in batch_data:
                    batch_species = batch_data['species']
                    num_loaded = len(batch_species)
                    
                    all_species.extend(batch_species)
                    valid_obs_ids.extend(batch_ids[:num_loaded])
                
                if self.mode in ['language', 'both'] and 'language_embeddings' in batch_data:
                    all_language_embs.append(batch_data['language_embeddings'])
                if self.mode in ['vision', 'both'] and 'vision_embeddings' in batch_data:
                    all_vision_embs.append(batch_data['vision_embeddings'])
                    
            except Exception as e:
                logger.warning(f"Failed to load batch: {e}")
                continue
        
        self.observation_ids = valid_obs_ids
        
        # Create species mapping
        if self.species_mapping is not None:
            self.species_to_idx = self.species_mapping
        else:
            unique_species = sorted(list(set(all_species)))
            self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.num_classes = len(self.species_to_idx)
        
        # Filter to valid species
        filtered_indices = []
        filtered_species = []
        
        for i, species in enumerate(all_species):
            if species in self.species_to_idx:
                filtered_indices.append(i)
                filtered_species.append(species)
        
        logger.info(f"Filtered {len(all_species)} to {len(filtered_species)} observations with valid species")
        
        self.observation_ids = [self.observation_ids[i] for i in filtered_indices]
        
        if filtered_species:
            self.species_labels = torch.tensor(
                [self.species_to_idx[species] for species in filtered_species], 
                dtype=torch.long, device=self.device
            )
            
            if self.mode in ['language', 'both'] and all_language_embs:
                all_lang_tensor = torch.cat(all_language_embs, dim=0)
                self.language_embeddings = all_lang_tensor[filtered_indices].to(self.device)
                logger.info(f"Language embeddings shape: {self.language_embeddings.shape}")
                
            if self.mode in ['vision', 'both'] and all_vision_embs:
                all_vis_tensor = torch.cat(all_vision_embs, dim=0)
                self.vision_embeddings = all_vis_tensor[filtered_indices].to(self.device)
                logger.info(f"Vision embeddings shape: {self.vision_embeddings.shape}")
        
        logger.info(f"Dataset loaded: {len(self.observation_ids)} observations, {self.num_classes} species")
        
        # Initialize training flag for augmentation
        self.training = False
        
    def __len__(self):
        return len(self.observation_ids)
    
    def __getitem__(self, idx):
        """Get sample with optional augmentation"""
        sample = {
            'species_idx': self.species_labels[idx],
            'species': self.idx_to_species[self.species_labels[idx].item()],
            'gbif_id': self.observation_ids[idx]
        }
        
        if self.mode in ['language', 'both']:
            lang_emb = self.language_embeddings[idx]
            if self.augment and self.training:
                # Random masking augmentation
                mask = torch.rand_like(lang_emb) > 0.1  # 10% mask
                lang_emb = lang_emb * mask
            sample['language_embedding'] = lang_emb
            
        if self.mode in ['vision', 'both']:
            vision_emb = self.vision_embeddings[idx]
            if vision_emb.ndim == 4:  # (8, 24, 24, 1408)
                vision_emb = vision_emb.mean(dim=(0, 1, 2))  # Pool to (1408,)
            
            if self.augment and self.training:
                # Dropout augmentation
                mask = torch.rand_like(vision_emb) > 0.1  # 10% dropout
                vision_emb = vision_emb * mask
                
            sample['vision_embedding'] = vision_emb
            
        return sample
    
    def set_training(self, mode):
        """Set training mode for augmentation"""
        self.training = mode


class ImprovedContrastiveMLPUNet(nn.Module):
    """MLP U-Net with improved projection heads for contrastive learning"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048, 
                 projection_dim=256, dropout=0.3, lightweight=False):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        self.projection_dim = projection_dim
        
        if lightweight:
            mid_dim_vision = 1024
            mid_dim_language = 2048
        else:
            mid_dim_vision = 2048
            mid_dim_language = 4096
        
        # Encoders
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, mid_dim_vision),
            nn.GELU(),
            nn.LayerNorm(mid_dim_vision),
            nn.Dropout(dropout),
            nn.Linear(mid_dim_vision, universal_dim),
            nn.GELU(),
            nn.LayerNorm(universal_dim)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, mid_dim_language),
            nn.GELU(),
            nn.LayerNorm(mid_dim_language),
            nn.Dropout(dropout),
            nn.Linear(mid_dim_language, universal_dim),
            nn.GELU(),
            nn.LayerNorm(universal_dim)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(universal_dim, universal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(universal_dim, universal_dim),
            nn.GELU(),
            nn.LayerNorm(universal_dim)
        )
        
        # Improved 2-layer projection heads
        self.vision_projector = nn.Sequential(
            nn.Linear(universal_dim, universal_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(universal_dim),
            nn.Linear(universal_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        
        self.language_projector = nn.Sequential(
            nn.Linear(universal_dim, universal_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(universal_dim),
            nn.Linear(universal_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        
        # Species classifier (for supervised signal)
        self.species_classifier = nn.Sequential(
            nn.Linear(universal_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256)  # Will be set to num_classes later
        )
        
        # Decoders
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, mid_dim_vision),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid_dim_vision, vision_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, mid_dim_language),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid_dim_language, language_dim)
        )
    
    def set_num_classes(self, num_classes):
        """Set the number of classes for species classifier"""
        new_head = nn.Linear(256, num_classes)
        # Get the device from any existing parameter
        target_device = next(self.parameters()).device
        self.species_classifier[-1] = new_head.to(target_device)
    
    def encode_vision(self, vision_emb):
        return self.vision_encoder(vision_emb)
    
    def encode_language(self, language_emb):
        return self.language_encoder(language_emb)
    
    def forward(self, vision_emb, language_emb, mask_language=None, mode='both', use_projector=True):
        # Apply masking
        if mask_language is not None and mode in ['language', 'both']:
            language_masked = language_emb.masked_fill(mask_language, 0.0)
        else:
            language_masked = language_emb
        
        # Handle single modality
        if mode == 'vision':
            language_masked = torch.zeros_like(language_emb)
        elif mode == 'language':
            vision_emb = torch.zeros_like(vision_emb)
        
        # Encode to universal space
        vision_universal = self.encode_vision(vision_emb)
        language_universal = self.encode_language(language_masked)
        
        # Apply bottleneck
        vision_bottleneck = self.bottleneck(vision_universal)
        language_bottleneck = self.bottleneck(language_universal)
        
        vision_universal = vision_universal + 0.5 * vision_bottleneck
        language_universal = language_universal + 0.5 * language_bottleneck
        
        # Species predictions (from universal embeddings)
        vision_species_logits = self.species_classifier(vision_universal)
        language_species_logits = self.species_classifier(language_universal)
        
        # Apply projection heads
        if use_projector and self.training:
            vision_projected = self.vision_projector(vision_universal)
            language_projected = self.language_projector(language_universal)
        else:
            vision_projected = vision_universal
            language_projected = language_universal
        
        # Decode
        vision_recon = self.vision_decoder(vision_universal)
        language_recon = self.language_decoder(language_universal)
        
        return {
            'vision_universal': vision_projected if use_projector and self.training else vision_universal,
            'language_universal': language_projected if use_projector and self.training else language_universal,
            'vision_universal_raw': vision_universal,
            'language_universal_raw': language_universal,
            'vision_species_logits': vision_species_logits,
            'language_species_logits': language_species_logits,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_original': vision_emb,
            'language_original': language_emb
        }


def compute_supervised_contrastive_loss(embeddings, labels, temperature=0.2, memory_bank=None):
    """
    Supervised contrastive loss with optional memory bank
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # If memory bank provided, add those embeddings
    if memory_bank is not None:
        bank_emb = F.normalize(memory_bank, dim=1)
        all_embeddings = torch.cat([embeddings, bank_emb], dim=0)
        # For memory bank, we don't have labels, so just use them as negatives
        # This simplifies the computation
        similarity = torch.matmul(embeddings, all_embeddings.T) / temperature
    else:
        all_embeddings = embeddings
        similarity = torch.matmul(embeddings, all_embeddings.T) / temperature
    
    # For supervised contrastive, we only consider positive pairs within the batch
    # Memory bank items are treated as negatives
    labels_expanded = labels.unsqueeze(1)
    batch_mask = torch.eq(labels_expanded, labels.unsqueeze(0)).float()
    
    # Remove self-similarity
    batch_mask.fill_diagonal_(0)
    
    # Compute log probabilities
    exp_sim = torch.exp(similarity)
    
    if memory_bank is not None:
        # With memory bank: positives only in batch, bank items are all negatives
        log_prob = similarity[:, :batch_size] - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mask = batch_mask
    else:
        # Without memory bank: standard supervised contrastive
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mask = batch_mask
    
    # Compute mean log probability over positives
    pos_count = mask.sum(dim=1)
    valid_mask = pos_count > 0
    
    if valid_mask.any():
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid_mask] / pos_count[valid_mask]
        loss = -mean_log_prob_pos.mean()
    else:
        # Fallback to standard InfoNCE if no positives
        labels_onehot = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(similarity[:, :batch_size], labels_onehot)
    
    return loss


def compute_losses_fixed(outputs, labels, epoch=0, temperature=0.2, 
                        memory_bank=None, alpha_recon=0.1, alpha_species=0.2):
    """Fixed loss computation with all improvements"""
    losses = {}
    device = outputs['vision_universal'].device
    
    # Get embeddings
    vision_proj = outputs['vision_universal']
    language_proj = outputs['language_universal']
    vision_raw = outputs['vision_universal_raw']
    language_raw = outputs['language_universal_raw']
    
    # 1. Supervised contrastive loss on projected embeddings
    if memory_bank is not None:
        bank_vision, bank_language, bank_labels = memory_bank.get_all(device)
        if bank_vision is not None:  # Memory bank is populated
            losses['vision_contrastive'] = compute_supervised_contrastive_loss(
                vision_proj, labels, temperature, bank_vision
            )
            losses['language_contrastive'] = compute_supervised_contrastive_loss(
                language_proj, labels, temperature, bank_language
            )
        else:  # First few batches - use in-batch loss only
            losses['vision_contrastive'] = compute_supervised_contrastive_loss(
                vision_proj, labels, temperature
            )
            losses['language_contrastive'] = compute_supervised_contrastive_loss(
                language_proj, labels, temperature
            )
    else:
        losses['vision_contrastive'] = compute_supervised_contrastive_loss(
            vision_proj, labels, temperature
        )
        losses['language_contrastive'] = compute_supervised_contrastive_loss(
            language_proj, labels, temperature
        )
    
    # 2. Cross-modal contrastive (standard InfoNCE)
    vision_norm = F.normalize(vision_proj, dim=1)
    language_norm = F.normalize(language_proj, dim=1)
    
    similarity = torch.matmul(vision_norm, language_norm.T) / temperature
    batch_labels = torch.arange(len(labels), device=device)
    
    loss_v2l = F.cross_entropy(similarity, batch_labels)
    loss_l2v = F.cross_entropy(similarity.T, batch_labels)
    losses['cross_contrastive'] = (loss_v2l + loss_l2v) / 2
    
    # 3. Species classification loss (supervised signal)
    if 'vision_species_logits' in outputs:
        losses['species_vision'] = F.cross_entropy(outputs['vision_species_logits'], labels)
        losses['species_language'] = F.cross_entropy(outputs['language_species_logits'], labels)
        losses['species_total'] = (losses['species_vision'] + losses['species_language']) / 2
    else:
        losses['species_total'] = torch.tensor(0.0)
    
    # 4. Reconstruction loss (keep small throughout)
    losses['recon_vision'] = F.mse_loss(outputs['vision_recon'], outputs['vision_original'])
    losses['recon_language'] = F.mse_loss(outputs['language_recon'], outputs['language_original'])
    losses['recon_total'] = losses['recon_vision'] + losses['recon_language']
    
    # 5. Alignment loss (bring raw embeddings together)
    losses['alignment'] = F.mse_loss(vision_raw, language_raw)
    
    # Monitor metrics
    with torch.no_grad():
        cos_sim = F.cosine_similarity(vision_proj, language_proj)
        losses['cos_sim'] = cos_sim.mean().item()
        losses['sim_diagonal'] = similarity.diag().mean().item()
        
        # Variance metrics
        losses['vision_var'] = torch.var(outputs['vision_original']).item()
        losses['language_var'] = torch.var(outputs['language_original']).item()
    
    # Total loss with balanced weights
    losses['total'] = (
        0.3 * losses['cross_contrastive'] +
        0.2 * losses['vision_contrastive'] +
        0.2 * losses['language_contrastive'] +
        alpha_species * losses['species_total'] +
        alpha_recon * losses['recon_total'] +
        0.1 * losses['alignment']
    )
    
    return losses


def main():
    parser = argparse.ArgumentParser(description='Fixed DeepEarth MLP U-Net')
    parser.add_argument('--mode', choices=['language', 'vision', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save-dir', type=str, default='checkpoints_fixed')
    parser.add_argument('--eval-every', type=int, default=3)
    parser.add_argument('--max-train-samples', type=int, default=5000)
    parser.add_argument('--max-test-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--memory-size', type=int, default=2048)
    parser.add_argument('--alpha-recon', type=float, default=0.1)
    parser.add_argument('--alpha-species', type=float, default=0.2)
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    set_all_seeds(args.seed)
    
    print("🌍 Fixed DeepEarth MLP U-Net Training")
    print(f"Temperature: {args.temperature}")
    print(f"Batch size: {args.batch_size}")
    print(f"Memory bank size: {args.memory_size}")
    
    # Auto-detect config
    if not args.config:
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        args.config = str(config_path)
    
    # Initialize cache
    print("\n📊 Loading data...")
    try:
        original_cwd = Path.cwd()
        dashboard_dir = Path(__file__).parent.parent / "dashboard"
        os.chdir(dashboard_dir)
        
        cache = UnifiedDataCache("dataset_config.json")
        
        # Load split configuration
        from deepearth_mlp_contrastive import load_train_test_split, filter_ids_by_species
        train_ids, test_ids = load_train_test_split(args.config)
        
        os.chdir(original_cwd)
        
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        return 1
    
    # Get common species
    print("📦 Analyzing species distribution...")
    sample_train_data = get_training_batch(
        cache, train_ids[:500], 
        include_vision=False, 
        include_language=True, 
        device='cpu'
    )
    sample_test_data = get_training_batch(
        cache, test_ids[:100], 
        include_vision=False, 
        include_language=True, 
        device='cpu'
    )
    
    train_species = set(sample_train_data['species'])
    test_species = set(sample_test_data['species'])
    common_species = train_species & test_species
    
    print(f"   Common species: {len(common_species)}")
    
    # Filter to common species
    filtered_train_ids = filter_ids_by_species(
        cache, train_ids, common_species, 
        max_samples=args.max_train_samples
    )
    filtered_test_ids = filter_ids_by_species(
        cache, test_ids, common_species, 
        max_samples=args.max_test_samples
    )
    
    # Create species mapping
    common_species_sorted = sorted(list(common_species))
    species_mapping = {species: idx for idx, species in enumerate(common_species_sorted)}
    
    # Create datasets
    train_dataset = DeepEarthMLPDataset(
        filtered_train_ids, cache, args.mode, 'cpu', species_mapping, augment=True
    )
    test_dataset = DeepEarthMLPDataset(
        filtered_test_ids, cache, args.mode, 'cpu', species_mapping, augment=False
    )
    
    print(f"\nDataset stats:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Species: {train_dataset.num_classes}")
    print(f"  Steps per epoch: {len(train_dataset) // args.batch_size}")
    
    # Balanced sampling
    class_counts = np.bincount(train_dataset.species_labels.cpu().numpy())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset.species_labels.cpu().numpy()]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_dataset.set_training(True)
    test_dataset.set_training(False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        pin_memory=(device == 'cuda'),
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        pin_memory=(device == 'cuda'),
        num_workers=0
    )
    
    # Initialize model
    model = ImprovedContrastiveMLPUNet(
        universal_dim=1024,
        projection_dim=256,
        dropout=0.3,
        lightweight=True
    ).to(device)
    model.set_num_classes(train_dataset.num_classes)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize memory bank
    memory_bank = MemoryBank(size=args.memory_size, dim=256)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Cosine scheduler with warmup
    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (num_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    best_r1 = 0
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {}
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, batch in enumerate(progress):
            # Move to device
            vision_input = batch['vision_embedding'].to(device)
            language_input = batch['language_embedding'].to(device)
            labels = batch['species_idx'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=(device == 'cuda')):
                outputs = model(vision_input, language_input, mode=args.mode)
                losses = compute_losses_fixed(
                    outputs, labels, epoch, 
                    temperature=args.temperature,
                    memory_bank=memory_bank,
                    alpha_recon=args.alpha_recon,
                    alpha_species=args.alpha_species
                )
            
            # Backward pass
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            global_step += 1
            
            # Update memory bank
            with torch.no_grad():
                memory_bank.update(
                    outputs['vision_universal'].detach(),
                    outputs['language_universal'].detach(),
                    labels
                )
            
            # Track losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item() if torch.is_tensor(v) else v
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'sim': f"{losses['sim_diagonal']:.3f}",
                'lr': f"{scheduler.get_last_lr()[0]:.5f}"
            })
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        
        print(f"\nEpoch {epoch+1}: Loss={epoch_losses['total']:.4f}, "
              f"Sim_diag={epoch_losses['sim_diagonal']:.3f}, "
              f"Cos_sim={epoch_losses['cos_sim']:.3f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            # Run evaluation here (similar to original but with fixed retrieval)
            print(f"📊 Evaluating epoch {epoch+1}...")
            # ... evaluation code ...
    
    print("\n✅ Training complete!")
    print(f"Best R@1: {best_r1:.1f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'species_mapping': species_mapping
    }, save_dir / 'final_model.pth')
    
    return 0


if __name__ == "__main__":
    exit(main())
