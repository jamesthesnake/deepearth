#!/usr/bin/env python3
"""
DeepEarth MLP U-Net Training with Contrastive Learning
Complete working version with all improvements
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

# Add dashboard to path for data access
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

# Setup logging
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


def load_train_test_split(config_path: str):
    """Load train/test split from configuration file."""
    logger.info(f"Loading train/test split from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    train_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                 if metadata['split'] == 'train']
    test_ids = [obs_id for obs_id, metadata in observation_mappings.items() 
                if metadata['split'] == 'test']
    
    logger.info(f"Loaded split: {len(train_ids)} train, {len(test_ids)} test observations")
    return train_ids, test_ids


def filter_ids_by_species(cache, obs_ids, target_species, max_samples=500):
    """Filter observation IDs to only include observations from target species."""
    filtered_ids = []
    batch_size = 64
    
    for i in range(0, min(len(obs_ids), max_samples * 2), batch_size):
        batch_ids = obs_ids[i:i + batch_size]
        if not batch_ids:
            break
            
        try:
            batch_data = get_training_batch(
                cache, batch_ids, 
                include_vision=False, 
                include_language=True, 
                device='cpu'
            )
            
            if 'species' in batch_data:
                for j, species in enumerate(batch_data['species']):
                    if species in target_species and len(filtered_ids) < max_samples:
                        if j < len(batch_ids):
                            filtered_ids.append(batch_ids[j])
                        
        except Exception as e:
            logger.warning(f"Failed to filter batch: {e}")
            continue
                
        if len(filtered_ids) >= max_samples:
            break
            
    return filtered_ids


class DeepEarthMLPDataset(Dataset):
    """Dataset that loads all data upfront"""
    
    def __init__(self, observation_ids, cache, mode='both', device='cpu', species_mapping=None):
        self.observation_ids = observation_ids
        self.cache = cache
        self.mode = mode
        self.device = 'cpu'  # Always load to CPU first for DataLoader
        self.species_mapping = species_mapping
        
        # Load all data at initialization
        self._load_dataset()
        
    def _load_dataset(self):
        """Load and prepare all data at initialization."""
        logger.info(f"Loading dataset with {len(self.observation_ids)} observations...")
        
        # Load data in batches to manage memory
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
                    device='cpu'  # Load to CPU first
                )
                
                # Debug: print keys on first batch
                if i == 0:
                    logger.info(f"Batch data keys: {list(batch_data.keys())}")
                
                # The batch data doesn't return observation_ids, so we use the input batch_ids
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
        
        # Update observation IDs to only valid ones
        self.observation_ids = valid_obs_ids
        
        # Create species label mapping
        if self.species_mapping is not None:
            self.species_to_idx = self.species_mapping
            self.idx_to_species = {idx: species for species, idx in self.species_mapping.items()}
            self.num_classes = len(self.species_mapping)
        else:
            unique_species = sorted(list(set(all_species)))
            self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
            self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
            self.num_classes = len(unique_species)
        
        # Filter species to only those in the mapping
        filtered_indices = []
        filtered_species = []
        
        for i, species in enumerate(all_species):
            if species in self.species_to_idx:
                filtered_indices.append(i)
                filtered_species.append(species)
        
        logger.info(f"Filtered {len(all_species)} to {len(filtered_species)} observations with valid species")
        
        # Update observation IDs to match filtered data
        self.observation_ids = [self.observation_ids[i] for i in filtered_indices]
        
        # Convert to tensors
        if filtered_species:  # Only if we have valid data
            self.species_labels = torch.tensor(
                [self.species_to_idx[species] for species in filtered_species], 
                dtype=torch.long, device=self.device
            )
            
            if self.mode in ['language', 'both'] and all_language_embs:
                # Concatenate all embeddings
                all_lang_tensor = torch.cat(all_language_embs, dim=0)
                # Filter to valid indices
                self.language_embeddings = all_lang_tensor[filtered_indices].to(self.device)
                logger.info(f"Language embeddings shape: {self.language_embeddings.shape}")
                
            if self.mode in ['vision', 'both'] and all_vision_embs:
                # Concatenate all embeddings
                all_vis_tensor = torch.cat(all_vision_embs, dim=0)
                # Filter to valid indices
                self.vision_embeddings = all_vis_tensor[filtered_indices].to(self.device)
                logger.info(f"Vision embeddings shape: {self.vision_embeddings.shape}")
        else:
            # No valid data loaded
            self.species_labels = torch.tensor([], dtype=torch.long, device=self.device)
            if self.mode in ['language', 'both']:
                self.language_embeddings = torch.empty((0, 7168), device=self.device)
            if self.mode in ['vision', 'both']:
                self.vision_embeddings = torch.empty((0, 8, 24, 24, 1408), device=self.device)
        
        logger.info(f"Dataset loaded: {len(self.observation_ids)} observations, {self.num_classes} species")
        
    def __len__(self):
        return len(self.observation_ids)
    
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
            # Vision embeddings need pooling from (8, 24, 24, 1408) to (1408,)
            vision_emb = self.vision_embeddings[idx]
            if vision_emb.ndim == 4:  # (8, 24, 24, 1408)
                vision_emb = vision_emb.mean(dim=(0, 1, 2))  # Pool to (1408,)
            sample['vision_embedding'] = vision_emb
            
        return sample


class ContrastiveMLPUNet(nn.Module):
    """MLP U-Net with contrastive learning projection heads"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048, 
                 projection_dim=256, dropout=0.3, lightweight=False):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        self.projection_dim = projection_dim
        
        if lightweight:
            # Lightweight version with fewer parameters
            mid_dim_vision = 1024
            mid_dim_language = 2048
        else:
            # Full version
            mid_dim_vision = 2048
            mid_dim_language = 4096
        
        # Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, mid_dim_vision),
            nn.GELU(),
            nn.LayerNorm(mid_dim_vision),
            nn.Dropout(dropout),
            nn.Linear(mid_dim_vision, universal_dim),
            nn.GELU(),
            nn.LayerNorm(universal_dim)
        )
        
        # Language encoder
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
        
        # Projection heads for contrastive learning
        self.vision_projector = nn.Sequential(
            nn.Linear(universal_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
        )
        
        self.language_projector = nn.Sequential(
            nn.Linear(universal_dim, projection_dim, bias=False),
            nn.LayerNorm(projection_dim)
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
    
    def encode_vision(self, vision_emb):
        """Encode vision embeddings to universal space"""
        return self.vision_encoder(vision_emb)
    
    def encode_language(self, language_emb):
        """Encode language embeddings to universal space"""
        return self.language_encoder(language_emb)
    
    def forward(self, vision_emb, language_emb, mask_language=None, mode='both', use_projector=True):
        """Forward pass with optional language masking and projection"""
        # Apply masking to language embeddings if provided
        if mask_language is not None and mode in ['language', 'both']:
            language_masked = language_emb.masked_fill(mask_language, 0.0)
        else:
            language_masked = language_emb
        
        # Handle single modality mode (for one-view alignment)
        if mode == 'vision':
            language_masked = torch.zeros_like(language_emb)
        elif mode == 'language':
            vision_emb = torch.zeros_like(vision_emb)
        
        # Encode to universal space
        vision_universal = self.encode_vision(vision_emb)
        language_universal = self.encode_language(language_masked)
        
        # Apply bottleneck with residual
        vision_bottleneck = self.bottleneck(vision_universal)
        language_bottleneck = self.bottleneck(language_universal)
        
        vision_universal = vision_universal + 0.5 * vision_bottleneck
        language_universal = language_universal + 0.5 * language_bottleneck
        
        # Apply projection heads for contrastive learning (training only)
        if use_projector and self.training:
            vision_projected = self.vision_projector(vision_universal)
            language_projected = self.language_projector(language_universal)
        else:
            vision_projected = vision_universal
            language_projected = language_universal
        
        # Decode back to original spaces
        vision_recon = self.vision_decoder(vision_universal)
        language_recon = self.language_decoder(language_universal)
        
        return {
            'vision_universal': vision_projected if use_projector and self.training else vision_universal,
            'language_universal': language_projected if use_projector and self.training else language_universal,
            'vision_universal_raw': vision_universal,  # Before projection
            'language_universal_raw': language_universal,  # Before projection
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_original': vision_emb,
            'language_original': language_emb
        }


def compute_losses(outputs, epoch=0, compute_alignment=True, freeze_decoders_epoch=5):
    """Compute reconstruction and contrastive losses with proper scaling"""
    losses = {}
    
    # Compute variances for baseline
    vision_var = torch.var(outputs['vision_original']).item()
    language_var = torch.var(outputs['language_original']).item()
    
    # L2/MSE reconstruction losses (only if decoders not frozen)
    if epoch < freeze_decoders_epoch:
        losses['recon_vision'] = F.mse_loss(outputs['vision_recon'], outputs['vision_original'])
        losses['recon_language'] = F.mse_loss(outputs['language_recon'], outputs['language_original'])
        losses['recon_total'] = losses['recon_vision'] + losses['recon_language']
    else:
        losses['recon_vision'] = torch.tensor(0.0)
        losses['recon_language'] = torch.tensor(0.0)
        losses['recon_total'] = torch.tensor(0.0)
    
    # Store variances for logging
    losses['vision_var'] = vision_var
    losses['language_var'] = language_var
    
    if compute_alignment:
        # Get universal embeddings
        vision_universal = outputs['vision_universal']
        language_universal = outputs['language_universal']
        
        # Normalize embeddings
        vision_norm = F.normalize(vision_universal, dim=1)
        language_norm = F.normalize(language_universal, dim=1)
        
        # Compute cosine similarity for monitoring
        cos_sim = F.cosine_similarity(vision_universal, language_universal)
        losses['cos_sim'] = cos_sim.mean().item()
        
        # InfoNCE/NT-Xent contrastive loss
        temperature = 0.07
        similarity_matrix = torch.matmul(vision_norm, language_norm.T) / temperature
        
        # Labels are diagonal (matching pairs)
        batch_size = vision_universal.shape[0]
        labels = torch.arange(batch_size, device=vision_universal.device)
        
        # Symmetric cross-entropy loss
        loss_v2l = F.cross_entropy(similarity_matrix, labels)
        loss_l2v = F.cross_entropy(similarity_matrix.T, labels)
        losses['contrastive'] = (loss_v2l + loss_l2v) / 2
        
        # Old alignment loss (for backward compatibility, but not used)
        losses['alignment'] = 1 - cos_sim.mean()
    else:
        losses['alignment'] = torch.tensor(0.0)
        losses['contrastive'] = torch.tensor(0.0)
        losses['cos_sim'] = 0.0
    
    # Total loss with new weighting
    if epoch < freeze_decoders_epoch:
        # Early epochs: balance reconstruction and contrastive
        losses['total'] = 0.3 * losses['recon_total'] + 0.7 * losses['contrastive']
    else:
        # After freezing decoders: pure contrastive
        losses['total'] = losses['contrastive']
    
    return losses


def evaluate_retrieval(model, dataloader, device):
    """Evaluate cross-modal retrieval performance"""
    model.eval()
    
    all_vision_emb = []
    all_language_emb = []
    all_species = []
    all_species_idx = []
    
    with torch.no_grad():
        for batch in dataloader:
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            
            # Use raw universal embeddings (before projection) for evaluation
            outputs = model(vision_emb, language_emb, use_projector=False)
            
            all_vision_emb.append(outputs['vision_universal'].cpu())
            all_language_emb.append(outputs['language_universal'].cpu())
            all_species.extend(batch['species'])
            all_species_idx.append(batch['species_idx'].cpu())
    
    # Concatenate all embeddings
    all_vision_emb = torch.cat(all_vision_emb, dim=0)
    all_language_emb = torch.cat(all_language_emb, dim=0)
    all_species_idx = torch.cat(all_species_idx, dim=0)
    
    # Normalize embeddings
    vision_norm = F.normalize(all_vision_emb, dim=1)
    language_norm = F.normalize(all_language_emb, dim=1)
    
    # Compute similarity matrix
    similarities = torch.matmul(vision_norm, language_norm.T)
    
    # Calculate retrieval metrics
    n = len(all_vision_emb)
    
    # Vision → Language retrieval
    v2l_r1 = 0
    v2l_r5 = 0
    v2l_r10 = 0
    
    for i in range(n):
        # Get top k
        top_k = torch.topk(similarities[i], k=min(n, 10), largest=True)
        top_k_indices = top_k.indices
        
        # Check if correct match is in top k
        correct_species = all_species_idx[i]
        retrieved_species = all_species_idx[top_k_indices]
        
        if correct_species in retrieved_species[:1]:
            v2l_r1 += 1
        if correct_species in retrieved_species[:5]:
            v2l_r5 += 1
        if correct_species in retrieved_species[:10]:
            v2l_r10 += 1
    
    # Language → Vision retrieval
    l2v_r1 = 0
    l2v_r5 = 0
    l2v_r10 = 0
    
    similarities_t = similarities.T
    for i in range(n):
        # Get top k
        top_k = torch.topk(similarities_t[i], k=min(n, 10), largest=True)
        top_k_indices = top_k.indices
        
        # Check if correct match is in top k
        correct_species = all_species_idx[i]
        retrieved_species = all_species_idx[top_k_indices]
        
        if correct_species in retrieved_species[:1]:
            l2v_r1 += 1
        if correct_species in retrieved_species[:5]:
            l2v_r5 += 1
        if correct_species in retrieved_species[:10]:
            l2v_r10 += 1
    
    results = {
        'v2l_r1': v2l_r1 / n,
        'v2l_r5': v2l_r5 / n,
        'v2l_r10': v2l_r10 / n,
        'l2v_r1': l2v_r1 / n,
        'l2v_r5': l2v_r5 / n,
        'l2v_r10': l2v_r10 / n,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='DeepEarth MLP U-Net with Contrastive Learning')
    parser.add_argument('--mode', choices=['language', 'vision', 'both'], default='both',
                       help='Training mode')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--save-dir', type=str, default='checkpoints_contrastive', help='Save directory')
    parser.add_argument('--mask-ratio', type=float, default=0.2, help='Language masking ratio')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--max-train-samples', type=int, default=2000, help='Maximum training samples')
    parser.add_argument('--max-test-samples', type=int, default=400, help='Maximum test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Path to train/test split config')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--universal-dim', type=int, default=2048, help='Universal embedding dimension')
    parser.add_argument('--projection-dim', type=int, default=256, help='Projection head dimension')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model version')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    set_all_seeds(args.seed)
    
    print("🌍 DeepEarth MLP U-Net with Contrastive Learning")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Universal dimension: {args.universal_dim}")
    print(f"Projection dimension: {args.projection_dim}")
    print(f"Dropout: {args.dropout}")
    
    # Auto-detect config file if not provided
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
        train_ids, test_ids = load_train_test_split(args.config)
        
        os.chdir(original_cwd)
        
    except Exception as e:
        logger.error(f"Failed to initialize cache: {e}")
        return 1
    
    # Analyze species distribution
    print("📦 Analyzing species distribution...")
    try:
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
        
        print(f"   Train species: {len(train_species)}, Test species: {len(test_species)}")
        print(f"   Common species: {len(common_species)}")
        
    except Exception as e:
        logger.error(f"Failed to analyze species: {e}")
        return 1
    
    if len(common_species) == 0:
        logger.error("No common species between train and test sets!")
        return 1
    
    # Filter to common species
    print("   Filtering to common species...")
    filtered_train_ids = filter_ids_by_species(
        cache, train_ids, common_species, 
        max_samples=args.max_train_samples
    )
    filtered_test_ids = filter_ids_by_species(
        cache, test_ids, common_species, 
        max_samples=args.max_test_samples
    )
    
    print(f"   Filtered train: {len(filtered_train_ids)}, test: {len(filtered_test_ids)}")
    
    # Create consistent species mapping
    common_species_sorted = sorted(list(common_species))
    species_mapping = {species: idx for idx, species in enumerate(common_species_sorted)}
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DeepEarthMLPDataset(
        filtered_train_ids, cache, args.mode, 'cpu', species_mapping
    )
    test_dataset = DeepEarthMLPDataset(
        filtered_test_ids, cache, args.mode, 'cpu', species_mapping
    )
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.error("No valid observations found in datasets!")
        return 1
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Total species: {train_dataset.num_classes}")
    print(f"  Avg samples per species: {len(train_dataset) / train_dataset.num_classes:.1f}")
    
    # Create dataloaders with weighted sampling for class balance
    print("\nCreating balanced dataloaders...")
    
    # Compute class weights for balanced sampling
    class_counts = np.bincount(train_dataset.species_labels.cpu().numpy())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_dataset.species_labels.cpu().numpy()]
    
    # Convert to tensor for WeightedRandomSampler
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    
    # Create weighted sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,  # Use weighted sampler instead of shuffle
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
    model = ContrastiveMLPUNet(
        universal_dim=args.universal_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        lightweight=args.lightweight
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.lightweight:
        print("  Using lightweight model version")
    
    # Training setup with improved scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # CSV logging
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'recon_loss', 'contrastive_loss', 
                        'cos_sim', 'vision_var', 'language_var',
                        'v2l_r1', 'v2l_r5', 'l2v_r1', 'l2v_r5'])
    
    # Compute baseline metrics
    print("\n📊 Computing baseline metrics...")
    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        vision_emb = sample_batch['vision_embedding'].to(device)
        language_emb = sample_batch['language_embedding'].to(device)
        
        # Compute variances
        vision_var = torch.var(vision_emb).item()
        language_var = torch.var(language_emb).item()
        
        # Random baseline for cosine similarity in universal space
        # First, pass through the model to get universal embeddings
        model.eval()
        outputs = model(vision_emb, language_emb)
        
        # Random baseline: what would random universal embeddings give?
        rand_universal_v = torch.randn_like(outputs['vision_universal'])
        rand_universal_l = torch.randn_like(outputs['language_universal'])
        rand_cos_sim = F.cosine_similarity(
            F.normalize(rand_universal_v, dim=1),
            F.normalize(rand_universal_l, dim=1)
        ).mean().item()
        
        # Also compute initial alignment of the model
        initial_cos_sim = F.cosine_similarity(
            outputs['vision_universal'],
            outputs['language_universal']
        ).mean().item()
        
        print(f"  Vision embedding variance: {vision_var:.4f}")
        print(f"  Language embedding variance: {language_var:.4f}")
        print(f"  Vision RMSE baseline: {np.sqrt(vision_var):.4f}")
        print(f"  Language RMSE baseline: {np.sqrt(language_var):.4f}")
        print(f"  Random universal cosine similarity: {rand_cos_sim:.4f}")
        print(f"  Initial model cosine similarity: {initial_cos_sim:.4f}")
        
        model.train()  # Switch back to training mode
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_r1 = 0
    freeze_decoders_epoch = 5
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'contrastive': 0, 'cos_sim': 0}
        epoch_vars = {'vision': 0, 'language': 0}
        
        # Freeze decoders after specified epoch
        if epoch == freeze_decoders_epoch:
            print(f"\n❄️ Freezing decoders at epoch {epoch+1}")
            for p in model.vision_decoder.parameters():
                p.requires_grad = False
            for p in model.language_decoder.parameters():
                p.requires_grad = False
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, batch in enumerate(progress):
            # Prepare inputs
            if args.mode in ['vision', 'both']:
                vision_input = batch['vision_embedding'].to(device)
            else:
                vision_input = torch.zeros(len(batch['species_idx']), 1408, device=device)
            
            if args.mode in ['language', 'both']:
                language_input = batch['language_embedding'].to(device)
                if args.mask_ratio > 0:
                    language_mask = (torch.rand_like(language_input) < args.mask_ratio)
                else:
                    language_mask = None
            else:
                language_input = torch.zeros(len(batch['species_idx']), 7168, device=device)
                language_mask = None
            
            # One-view alignment: randomly use only one modality 50% of the time
            training_mode = args.mode
            if args.mode == 'both' and np.random.rand() < 0.5:
                if np.random.rand() < 0.5:
                    training_mode = 'vision'
                else:
                    training_mode = 'language'
            
            # Forward pass with AMP
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                outputs = model(
                    vision_input,
                    language_input,
                    mask_language=language_mask,
                    mode=training_mode
                )
                losses = compute_losses(outputs, epoch=epoch, freeze_decoders_epoch=freeze_decoders_epoch)
            
            # Backward pass
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['recon'] += losses['recon_total'].item()
            epoch_losses['contrastive'] += losses.get('contrastive', torch.tensor(0.0)).item()
            epoch_losses['cos_sim'] += losses['cos_sim']
            epoch_vars['vision'] += losses['vision_var']
            epoch_vars['language'] += losses['language_var']
            
            progress.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'cos_sim': f"{losses['cos_sim']:.3f}"
            })
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_vars:
            epoch_vars[key] /= num_batches
        
        # Evaluate retrieval performance
        if (epoch + 1) % args.eval_every == 0:
            print(f"\n📊 Evaluating epoch {epoch+1}...")
            retrieval_results = evaluate_retrieval(model, test_loader, device)
            
            print(f"  Loss: {epoch_losses['total']:.4f} "
                  f"(Recon: {epoch_losses['recon']:.4f}, Contrastive: {epoch_losses['contrastive']:.4f})")
            print(f"  Avg cosine similarity: {epoch_losses['cos_sim']:.3f}")
            print(f"  Vision → Language: R@1={retrieval_results['v2l_r1']:.3f}, "
                  f"R@5={retrieval_results['v2l_r5']:.3f}")
            print(f"  Language → Vision: R@1={retrieval_results['l2v_r1']:.3f}, "
                  f"R@5={retrieval_results['l2v_r5']:.3f}")
            
            # Update learning rate based on validation loss
            scheduler.step(epoch_losses['total'])
            
            # Save best model
            avg_r1 = (retrieval_results['v2l_r1'] + retrieval_results['l2v_r1']) / 2
            if avg_r1 > best_r1:
                best_r1 = avg_r1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'args': vars(args),
                    'species_mapping': species_mapping,
                    'best_r1': best_r1
                }, save_dir / 'best_model.pth')
                print(f"  💾 New best model saved! (Avg R@1: {best_r1:.3f})")
            
            # Log to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, epoch_losses['total'], epoch_losses['recon'], 
                    epoch_losses['contrastive'], epoch_losses['cos_sim'],
                    epoch_vars['vision'], epoch_vars['language'],
                    retrieval_results['v2l_r1'], retrieval_results['v2l_r5'],
                    retrieval_results['l2v_r1'], retrieval_results['l2v_r5']
                ])
        else:
            # Still update scheduler on non-eval epochs
            scheduler.step(epoch_losses['total'])
            
            print(f"Epoch {epoch+1:2d}/{args.epochs}: "
                  f"Loss: {epoch_losses['total']:.4f}, "
                  f"Contrastive: {epoch_losses['contrastive']:.4f}, "
                  f"Cos Sim: {epoch_losses['cos_sim']:.3f}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'species_mapping': species_mapping,
        'args': vars(args)
    }, save_dir / 'final_model.pth')
    
    print("\n✅ Training complete!")
    print(f"Best avg R@1: {best_r1:.3f}")
    print(f"Results saved to: {args.save_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
