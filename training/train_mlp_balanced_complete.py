#!/usr/bin/env python3
"""
Enhanced Balanced MLP U-Net Training - With advanced retrieval optimizations
Implements: MoCo queue, temperature scheduling, bottleneck projections, hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import os
from tqdm import tqdm
import json
import math
from datetime import datetime
from collections import defaultdict, deque
import random
import faiss

# Add paths
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MoCoQueue:
    """Memory bank for MoCo-style negative sampling"""
    def __init__(self, dim=256, size=8192, device='cuda'):
        self.size = size
        self.dim = dim
        self.device = device
        
        # Initialize queues
        self.vision_queue = torch.randn(dim, size).to(device)
        self.language_queue = torch.randn(dim, size).to(device)
        self.ptr = 0
        
        # Normalize
        self.vision_queue = F.normalize(self.vision_queue, dim=0)
        self.language_queue = F.normalize(self.language_queue, dim=0)
        
    @torch.no_grad()
    def update(self, vision_emb, language_emb):
        """Update queue with new embeddings"""
        batch_size = vision_emb.shape[0]
        
        assert self.size % batch_size == 0, f"Queue size {self.size} must be divisible by batch size {batch_size}"
        
        # Replace oldest with newest
        self.vision_queue[:, self.ptr:self.ptr + batch_size] = vision_emb.T
        self.language_queue[:, self.ptr:self.ptr + batch_size] = language_emb.T
        
        self.ptr = (self.ptr + batch_size) % self.size
        
    def get_queue(self):
        """Get current queue tensors"""
        return self.vision_queue.T, self.language_queue.T


class DeepEarthDataset(Dataset):
    """Dataset with optional hard negative sampling"""
    
    def __init__(self, observation_ids, cache, species_mapping=None, mode='both', 
                 batch_load_size=64, hard_negatives_file=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.mode = mode
        self.species_mapping = species_mapping
        self.hard_negatives = None
        
        # Load hard negatives if provided
        if hard_negatives_file and os.path.exists(hard_negatives_file):
            logger.info(f"Loading hard negatives from {hard_negatives_file}")
            self.hard_negatives = torch.load(hard_negatives_file)
        
        # Load all data
        self._load_dataset(batch_load_size)
        
    def _load_dataset(self, batch_size):
        """Load dataset in batches with fp16 storage"""
        logger.info(f"Loading dataset with {len(self.observation_ids)} observations...")
        
        all_species = []
        all_language_embs = []
        all_vision_embs = []
        valid_obs_ids = []
        
        # Load in batches
        for i in range(0, len(self.observation_ids), batch_size):
            if i % 500 == 0:
                logger.info(f"  Loaded {i}/{len(self.observation_ids)} observations...")
                
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
                        all_language_embs.append(batch_data['language_embeddings'].half())
                        
                    if self.mode in ['vision', 'both'] and 'vision_embeddings' in batch_data:
                        vis_emb = batch_data['vision_embeddings']
                        if vis_emb.ndim == 5:  # (batch, 8, 24, 24, 1408)
                            vis_emb = vis_emb.mean(dim=(1, 2, 3))
                        elif vis_emb.ndim == 4:  # Single sample
                            vis_emb = vis_emb.mean(dim=(0, 1, 2)).unsqueeze(0)
                        all_vision_embs.append(vis_emb.half())
                        
            except Exception as e:
                logger.warning(f"Failed to load batch: {e}")
                continue
        
        # Create mappings and filter
        self.observation_ids = valid_obs_ids
        
        if self.species_mapping is None:
            unique_species = sorted(list(set(all_species)))
            self.species_mapping = {species: idx for idx, species in enumerate(unique_species)}
        
        self.species_to_idx = self.species_mapping
        self.idx_to_species = {idx: species for species, idx in self.species_mapping.items()}
        self.num_classes = len(self.species_mapping)
        
        # Filter to valid species
        filtered_indices = []
        filtered_species = []
        for i, species in enumerate(all_species):
            if species in self.species_to_idx:
                filtered_indices.append(i)
                filtered_species.append(species)
        
        # Create tensors
        self.species_labels = torch.tensor(
            [self.species_to_idx[species] for species in filtered_species], 
            dtype=torch.long
        )
        
        if self.mode in ['language', 'both'] and all_language_embs:
            self.language_embeddings = torch.cat(all_language_embs, dim=0)[filtered_indices]
            
        if self.mode in ['vision', 'both'] and all_vision_embs:
            self.vision_embeddings = torch.cat(all_vision_embs, dim=0)[filtered_indices]
            
        self.observation_ids = [self.observation_ids[i] for i in filtered_indices]
        
        logger.info(f"Dataset loaded: {len(self)} observations, {self.num_classes} species")
        
    def __len__(self):
        return len(self.species_labels)
    
    def __getitem__(self, idx):
        # Use hard negative sampling 50% of the time if available
        if self.hard_negatives and random.random() < 0.5:
            # Get a hard negative pair
            hard_idx = random.choice(self.hard_negatives.get(idx, [idx]))
            idx = hard_idx
            
        sample = {
            'species_idx': self.species_labels[idx],
            'species': self.idx_to_species[self.species_labels[idx].item()],
            'obs_id': self.observation_ids[idx],
            'idx': idx  # Add index for hard negative mining
        }
        
        # Convert to float32 and add augmentation
        if hasattr(self, 'language_embeddings'):
            lang_emb = self.language_embeddings[idx].float()
            # Add small noise during training
            if hasattr(self, 'training') and self.training:
                lang_emb = lang_emb + 0.01 * torch.randn_like(lang_emb)
            sample['language_embedding'] = lang_emb
        else:
            sample['language_embedding'] = torch.zeros(7168)
            
        if hasattr(self, 'vision_embeddings'):
            vision_emb = self.vision_embeddings[idx].float()
            # Add small noise during training
            if hasattr(self, 'training') and self.training:
                vision_emb = vision_emb + 0.01 * torch.randn_like(vision_emb)
            sample['vision_embedding'] = vision_emb
        else:
            sample['vision_embedding'] = torch.zeros(1408)
            
        return sample


class EnhancedMLPUNet(nn.Module):
    """MLP U-Net with bottleneck projections for better retrieval"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, hidden_dim=2048, 
                 bottleneck_dim=512, projection_dim=256, num_classes=10):
        super().__init__()
        
        # Encoders with larger hidden dimension
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Bottleneck + projection heads for contrastive learning
        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, projection_dim),
            nn.LayerNorm(projection_dim, elementwise_affine=False)
        )
        
        self.language_proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, projection_dim),
            nn.LayerNorm(projection_dim, elementwise_affine=False)
        )
        
        # Species classifier (uses hidden features, not bottlenecked)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, vision_emb, language_emb, return_features=False):
        # Encode
        vision_feat = self.vision_encoder(vision_emb)
        language_feat = self.language_encoder(language_emb)
        
        # Project through bottleneck for contrastive learning
        vision_proj = F.normalize(self.vision_proj(vision_feat), dim=1)
        language_proj = F.normalize(self.language_proj(language_feat), dim=1)
        
        # Classify from full features (not bottlenecked)
        combined = torch.cat([vision_feat, language_feat], dim=1)
        species_logits = self.classifier(combined)
        
        outputs = {
            'vision_proj': vision_proj,
            'language_proj': language_proj,
            'species_logits': species_logits
        }
        
        if return_features:
            outputs['vision_feat'] = vision_feat
            outputs['language_feat'] = language_feat
            
        return outputs


def temperature_schedule(epoch, T0=0.1, Tmin=0.03, epochs=30):
    """Cosine temperature schedule"""
    return Tmin + 0.5 * (T0 - Tmin) * (1 + math.cos(math.pi * epoch / epochs))


def compute_losses_with_queue(outputs, labels, moco_queue=None, temperature=0.1, 
                             alpha_cls=1.0, alpha_cont=1.0):
    """Compute losses with MoCo-style queue"""
    losses = {}
    device = outputs['vision_proj'].device
    batch_size = outputs['vision_proj'].shape[0]
    
    # Get projections
    vision_proj = outputs['vision_proj']
    language_proj = outputs['language_proj']
    
    if moco_queue is not None:
        # Get queue
        vision_queue, language_queue = moco_queue.get_queue()
        
        # Compute similarities with queue using matrix multiplication
        # V->L: vision queries against language keys (batch + queue)
        l_pos = torch.einsum('nc,nc->n', [vision_proj, language_proj]).unsqueeze(-1)
        l_neg_batch = vision_proj @ language_proj.T
        l_neg_queue = vision_proj @ language_queue.T
        
        # Remove self from batch negatives
        mask = torch.eye(batch_size, device=device).bool()
        l_neg_batch = l_neg_batch.masked_fill(mask, -65504.0)  # Max negative value for fp16
        
        # Combine positives and negatives
        logits_v2l = torch.cat([l_pos, l_neg_batch, l_neg_queue], dim=1) / temperature
        
        # L->V: language queries against vision keys
        v_pos = torch.einsum('nc,nc->n', [language_proj, vision_proj]).unsqueeze(-1)
        v_neg_batch = language_proj @ vision_proj.T
        v_neg_queue = language_proj @ vision_queue.T
        
        v_neg_batch = v_neg_batch.masked_fill(mask, -65504.0)  # Max negative value for fp16
        logits_l2v = torch.cat([v_pos, v_neg_batch, v_neg_queue], dim=1) / temperature
        
        # Labels are always 0 (positive is first)
        labels_cont = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        loss_v2l = F.cross_entropy(logits_v2l, labels_cont)
        loss_l2v = F.cross_entropy(logits_l2v, labels_cont)
        
    else:
        # Standard in-batch negatives only
        logits = torch.matmul(vision_proj, language_proj.T) / temperature
        
        # DO NOT mask diagonal - it contains the positive pairs!
        labels_cont = torch.arange(batch_size, device=device)
        
        loss_v2l = F.cross_entropy(logits, labels_cont)
        loss_l2v = F.cross_entropy(logits.T, labels_cont)
    
    losses['contrastive'] = (loss_v2l + loss_l2v) / 2
    
    # Classification loss
    losses['classification'] = F.cross_entropy(outputs['species_logits'], labels)
    
    # Total loss
    losses['total'] = alpha_cont * losses['contrastive'] + alpha_cls * losses['classification']
    
    # Metrics
    with torch.no_grad():
        # Classification accuracy
        pred = outputs['species_logits'].argmax(dim=1)
        losses['acc'] = (pred == labels).float().mean().item()
        
        # Top-5 accuracy
        top5 = outputs['species_logits'].topk(5, 1).indices
        losses['acc_top5'] = (top5 == labels[:, None]).any(1).float().mean().item()
        
        # Embedding statistics
        losses['vision_std'] = torch.std(vision_proj, dim=0).mean().item()
        losses['language_std'] = torch.std(language_proj, dim=0).mean().item()
        
        # Individual losses
        losses['loss_cont'] = losses['contrastive'].item()
        losses['loss_cls'] = losses['classification'].item()
        
    return losses


def mine_hard_negatives(model, dataset, device, k=10):
    """Mine hard negatives using FAISS"""
    model.eval()
    
    # Get all embeddings
    all_vision = []
    all_language = []
    all_labels = []
    all_indices = []
    
    # Create temporary dataloader
    temp_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Mining hard negatives")):
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            
            outputs = model(vision_emb, language_emb)
            
            all_vision.append(outputs['vision_proj'].cpu().numpy())
            all_language.append(outputs['language_proj'].cpu().numpy())
            all_labels.append(batch['species_idx'])
            
            # Use actual indices from batch
            if 'idx' in batch:
                all_indices.extend(batch['idx'].tolist())
            else:
                # Fallback: compute indices based on batch position
                batch_size = len(batch['species_idx'])
                start_idx = batch_idx * 128
                all_indices.extend(range(start_idx, start_idx + batch_size))
    
    # Concatenate
    all_vision = np.vstack(all_vision).astype('float32')
    all_language = np.vstack(all_language).astype('float32')
    all_labels = torch.cat(all_labels).numpy()
    
    # Build FAISS indices
    d = all_vision.shape[1]
    vision_index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
    language_index = faiss.IndexFlatIP(d)
    
    vision_index.add(all_vision)
    language_index.add(all_language)
    
    # Find hard negatives
    hard_negatives = {}
    
    for i in tqdm(range(len(all_indices)), desc="Finding hard negatives"):
        # V->L hard negatives
        _, v2l_indices = language_index.search(all_vision[i:i+1], k+1)
        v2l_hard = [idx for idx in v2l_indices[0] if all_labels[idx] != all_labels[i]][:k//2]
        
        # L->V hard negatives  
        _, l2v_indices = vision_index.search(all_language[i:i+1], k+1)
        l2v_hard = [idx for idx in l2v_indices[0] if all_labels[idx] != all_labels[i]][:k//2]
        
        hard_negatives[all_indices[i]] = v2l_hard + l2v_hard
    
    return hard_negatives


def evaluate(model, dataloader, device, temperature=0.1):
    """Evaluate with detailed metrics"""
    model.eval()
    
    metrics = defaultdict(float)
    num_batches = 0
    
    all_vision_proj = []
    all_language_proj = []
    all_labels = []
    all_species_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            labels = batch['species_idx'].to(device)
            
            outputs = model(vision_emb, language_emb)
            losses = compute_losses_with_queue(outputs, labels, temperature=temperature)
            
            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    metrics[k] += v
            
            all_vision_proj.append(outputs['vision_proj'])
            all_language_proj.append(outputs['language_proj'])
            all_labels.append(labels)
            all_species_preds.append(outputs['species_logits'].argmax(dim=1))
            
            num_batches += 1
    
    # Average metrics
    for k in metrics:
        metrics[k] /= num_batches
    
    # Global retrieval
    all_vision = torch.cat(all_vision_proj)
    all_language = torch.cat(all_language_proj)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_species_preds)
    
    similarities = torch.matmul(all_vision, all_language.T)
    
    # Compute retrieval metrics
    n = len(all_labels)
    v2l_r1 = v2l_r5 = l2v_r1 = l2v_r5 = 0
    
    for i in range(n):
        # V->L
        _, top5 = similarities[i].topk(min(5, n))
        if all_labels[i] == all_labels[top5[0]]:
            v2l_r1 += 1
        if any(all_labels[i] == all_labels[idx] for idx in top5):
            v2l_r5 += 1
        
        # L->V  
        _, top5 = similarities[:, i].topk(min(5, n))
        if all_labels[i] == all_labels[top5[0]]:
            l2v_r1 += 1
        if any(all_labels[i] == all_labels[idx] for idx in top5):
            l2v_r5 += 1
    
    metrics['v2l_r1'] = v2l_r1 / n
    metrics['l2v_r1'] = l2v_r1 / n
    metrics['v2l_r5'] = v2l_r5 / n
    metrics['l2v_r5'] = l2v_r5 / n
    metrics['avg_r1'] = (metrics['v2l_r1'] + metrics['l2v_r1']) / 2
    metrics['avg_r5'] = (metrics['v2l_r5'] + metrics['l2v_r5']) / 2
    
    return metrics


def load_split(config_path):
    """Load train/test split from config"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_ids = []
    test_ids = []
    
    for obs_id, meta in config['observation_mappings'].items():
        if meta['split'] == 'train':
            train_ids.append(obs_id)
        else:
            test_ids.append(obs_id)
    
    return train_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default='checkpoints_enhanced')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--alpha-cls', type=float, default=0.5)
    parser.add_argument('--alpha-cont', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--queue-size', type=int, default=8192)
    parser.add_argument('--temp-init', type=float, default=0.1)
    parser.add_argument('--temp-min', type=float, default=0.03)
    parser.add_argument('--hidden-dim', type=int, default=2048, help='Hidden dimension size')
    parser.add_argument('--mine-every', type=int, default=5, help='Mine hard negatives every N epochs')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("🌍 Enhanced Balanced MLP U-Net Training")
    print(f"Device: {device}")
    print(f"Queue size: {args.queue_size}")
    print(f"Temperature: {args.temp_init} → {args.temp_min}")
    
    # Initialize
    os.chdir(dashboard_path)
    cache = UnifiedDataCache("dataset_config.json")
    
    train_ids, test_ids = load_split(args.config)
    print(f"Loaded {len(train_ids)} train, {len(test_ids)} test IDs")
    
    if args.max_samples:
        train_ids = train_ids[:int(args.max_samples * 0.8)]
        test_ids = test_ids[:int(args.max_samples * 0.2)]
    
    # Create datasets
    hard_negatives_file = Path(args.save_dir) / 'hard_negatives.pth'
    train_dataset = DeepEarthDataset(
        train_ids, cache, 
        hard_negatives_file=hard_negatives_file if hard_negatives_file.exists() else None
    )
    test_dataset = DeepEarthDataset(
        test_ids, cache, 
        species_mapping=train_dataset.species_mapping
    )
    
    train_dataset.training = True
    test_dataset.training = False
    
    print(f"Train: {len(train_dataset)} samples, {train_dataset.num_classes} species")
    print(f"Test: {len(test_dataset)} samples")
    
    # Adjust queue size to be divisible by batch size
    if args.queue_size > 0:
        queue_size = (args.queue_size // args.batch_size) * args.batch_size
        if queue_size == 0:
            print(f"⚠️ Warning: Queue size {args.queue_size} is smaller than batch size {args.batch_size}")
            print("   Disabling MoCo queue...")
            moco_queue = None
        else:
            print(f"Adjusted queue size: {queue_size}")
            moco_queue = MoCoQueue(dim=256, size=queue_size, device=device)
    else:
        print("MoCo queue disabled (queue_size=0)")
        moco_queue = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = EnhancedMLPUNet(
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training
    Path(args.save_dir).mkdir(exist_ok=True)
    best_r1 = 0
    
    print("\n🚀 Starting training...")
    
    try:
        for epoch in range(args.epochs):
            # Mine hard negatives periodically
            if epoch > 0 and epoch % args.mine_every == 0:
                print(f"\n⛏️ Mining hard negatives at epoch {epoch}...")
                hard_negatives = mine_hard_negatives(model, train_dataset, device)
                torch.save(hard_negatives, hard_negatives_file)
                train_dataset.hard_negatives = hard_negatives
                print(f"Saved {len(hard_negatives)} hard negative mappings")
            
            # Temperature schedule
            temperature = temperature_schedule(epoch, args.temp_init, args.temp_min, args.epochs)
            
            # Train
            model.train()
            train_metrics = defaultdict(float)
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            for batch in pbar:
                vision_emb = batch['vision_embedding'].to(device)
                language_emb = batch['language_embedding'].to(device)
                labels = batch['species_idx'].to(device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = model(vision_emb, language_emb)
                    losses = compute_losses_with_queue(
                        outputs, labels, moco_queue, temperature,
                        args.alpha_cls, args.alpha_cont
                    )
                
                scaler.scale(losses['total']).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Synchronize before queue update to avoid stale data
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Update queue
                if moco_queue is not None:
                    with torch.no_grad():
                        moco_queue.update(outputs['vision_proj'], outputs['language_proj'])
                
                # Track metrics
                for k, v in losses.items():
                    if isinstance(v, (int, float)):
                        train_metrics[k] += v
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'acc': f"{losses['acc']:.2%}",
                    'T': f"{temperature:.3f}",
                    'σ_v': f"{losses['vision_std']:.3f}"
                })
            
            scheduler.step()
            
            # Average metrics
            for k in train_metrics:
                train_metrics[k] /= num_batches
            
            # Evaluate
            if (epoch + 1) % args.eval_every == 0:
                test_metrics = evaluate(model, test_loader, device, temperature)
                
                print(f"\nEpoch {epoch+1}:")
                print(f"  Train - Loss: {train_metrics['total']:.3f}, "
                      f"Cont: {train_metrics['loss_cont']:.3f}, "
                      f"Cls: {train_metrics['loss_cls']:.3f}, "
                      f"Acc: {train_metrics['acc']:.2%}")
                print(f"  Test  - Acc: {test_metrics['acc']:.2%}, "
                      f"Top5: {test_metrics['acc_top5']:.2%}")
                print(f"  Retrieval R@1: {test_metrics['avg_r1']:.2%} "
                      f"(V→L: {test_metrics['v2l_r1']:.2%}, L→V: {test_metrics['l2v_r1']:.2%})")
                print(f"  Retrieval R@5: {test_metrics['avg_r5']:.2%} "
                      f"(V→L: {test_metrics['v2l_r5']:.2%}, L→V: {test_metrics['l2v_r5']:.2%})")
                print(f"  Embedding std - Vision: {test_metrics['vision_std']:.3f}, "
                      f"Language: {test_metrics['language_std']:.3f}")
                
                # Save best
                if test_metrics['avg_r1'] > best_r1:
                    best_r1 = test_metrics['avg_r1']
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = Path(args.save_dir) / f'best_{ts}.pth'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'test_metrics': test_metrics,
                        'args': vars(args)
                    }, save_path)
                    print(f"  💾 New best R@1: {best_r1:.2%}")
                    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    
    print(f"\n✅ Best R@1: {best_r1:.2%}")


if __name__ == "__main__":
    main()
