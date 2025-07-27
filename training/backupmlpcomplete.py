#!/usr/bin/env python3
"""
Balanced MLP U-Net Training - Optimized for both retrieval and classification
Fixed version with proper species mapping and debugging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add paths
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

# Import from the working fixed version
from deepearth_simple_mlp import (
    DeepEarthMLPDataset, 
    MemoryBank,
    set_all_seeds,
    compute_supervised_contrastive_loss
)
from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BalancedMLPUNet(nn.Module):
    """MLP U-Net optimized for both retrieval AND classification"""
    
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048, 
                 projection_dim=256, dropout=0.3, num_classes=10):
        super().__init__()
        
        # Use lightweight dimensions
        mid_dim_vision = 1024
        mid_dim_language = 2048
        
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
        
        # Contrastive projectors (for retrieval)
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
        
        # Dedicated species classifiers (deeper, separate)
        self.species_head = nn.Sequential(
            nn.Linear(universal_dim * 2, 512),  # Takes concatenated embeddings
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
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
        
    def encode_vision(self, x):
        return self.vision_encoder(x)
    
    def encode_language(self, x):
        return self.language_encoder(x)
    
    def forward(self, vision_emb, language_emb, detach_species=False):
        # Encode
        vision_universal = self.encode_vision(vision_emb)
        language_universal = self.encode_language(language_emb)
        
        # Bottleneck
        vision_bottleneck = self.bottleneck(vision_universal)
        language_bottleneck = self.bottleneck(language_universal)
        
        vision_universal = vision_universal + 0.5 * vision_bottleneck
        language_universal = language_universal + 0.5 * language_bottleneck
        
        # Species classification (with optional detach)
        fusion_input = torch.cat([vision_universal, language_universal], dim=1)
        if detach_species:
            fusion_input = fusion_input.detach()
        species_logits = self.species_head(fusion_input)
        
        # Contrastive projections
        if self.training:
            vision_projected = self.vision_projector(vision_universal)
            language_projected = self.language_projector(language_universal)
        else:
            vision_projected = vision_universal
            language_projected = language_universal
        
        # Decode
        vision_recon = self.vision_decoder(vision_universal)
        language_recon = self.language_decoder(language_universal)
        
        return {
            'vision_universal': vision_projected,
            'language_universal': language_projected,
            'vision_universal_raw': vision_universal,
            'language_universal_raw': language_universal,
            'species_logits': species_logits,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_original': vision_emb,
            'language_original': language_emb
        }


def compute_balanced_losses(outputs, labels, epoch, temperature=0.2, 
                           alpha_species_max=0.5, memory_bank=None, debug=False):
    """Compute losses with species weight scheduling"""
    losses = {}
    device = outputs['vision_universal'].device
    
    # Species weight schedule (sigmoid ramp)
    alpha_species = alpha_species_max * (1 + np.tanh((epoch - 5) / 3)) / 2
    
    # 1. Contrastive losses
    vision_proj = outputs['vision_universal']
    language_proj = outputs['language_universal']
    
    if memory_bank is not None:
        bank_vision, bank_language, _ = memory_bank.get_all(device)
        if bank_vision is not None:
            losses['contrastive_v'] = compute_supervised_contrastive_loss(
                vision_proj, labels, temperature, bank_vision
            )
            losses['contrastive_l'] = compute_supervised_contrastive_loss(
                language_proj, labels, temperature, bank_language
            )
        else:
            losses['contrastive_v'] = compute_supervised_contrastive_loss(
                vision_proj, labels, temperature
            )
            losses['contrastive_l'] = compute_supervised_contrastive_loss(
                language_proj, labels, temperature
            )
    
    # Cross-modal contrastive
    vision_norm = F.normalize(vision_proj, dim=1)
    language_norm = F.normalize(language_proj, dim=1)
    sim = torch.matmul(vision_norm, language_norm.T) / temperature
    batch_labels = torch.arange(len(labels), device=device)
    
    losses['contrastive_cross'] = (F.cross_entropy(sim, batch_labels) + 
                                   F.cross_entropy(sim.T, batch_labels)) / 2
    
    # 2. Species classification
    losses['species'] = F.cross_entropy(outputs['species_logits'], labels)
    
    # 3. Reconstruction (small weight)
    losses['recon_v'] = F.mse_loss(outputs['vision_recon'], outputs['vision_original'])
    losses['recon_l'] = F.mse_loss(outputs['language_recon'], outputs['language_original'])
    losses['recon'] = losses['recon_v'] + losses['recon_l']
    
    # 4. Alignment
    losses['align'] = F.mse_loss(outputs['vision_universal_raw'], 
                                 outputs['language_universal_raw'])
    
    # Monitor metrics
    with torch.no_grad():
        cos_sim = F.cosine_similarity(vision_proj, language_proj).mean()
        losses['cos_sim'] = cos_sim.item()
        losses['sim_diag'] = sim.diag().mean().item()
        
        # Species accuracy (top-1 and top-5)
        species_pred = outputs['species_logits'].argmax(dim=1)
        species_acc = (species_pred == labels).float().mean()
        losses['species_acc'] = species_acc.item()
        
        # Top-5 accuracy
        _, top5_pred = outputs['species_logits'].topk(5, dim=1)
        top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))
        top5_acc = top5_correct.any(dim=1).float().mean()
        losses['species_acc_top5'] = top5_acc.item()
        
        # Debug: check prediction diversity
        if debug:
            unique_preds = len(torch.unique(species_pred))
            unique_labels = len(torch.unique(labels))
            print(f"Debug - Unique predictions: {unique_preds}/{unique_labels} unique labels")
            print(f"Debug - Label range: {labels.min().item()}-{labels.max().item()}")
            print(f"Debug - Prediction range: {species_pred.min().item()}-{species_pred.max().item()}")
    
    # Total loss
    losses['total'] = (
        0.25 * losses['contrastive_cross'] +
        0.15 * losses['contrastive_v'] +
        0.15 * losses['contrastive_l'] +
        alpha_species * losses['species'] +
        0.05 * losses['recon'] +
        0.05 * losses['align']
    )
    
    losses['alpha_species'] = alpha_species
    
    return losses


def evaluate_model_with_genus(model, test_loader, device, species_to_genus, genus_mapping):
    """Evaluate with species, genus-level accuracy, and retrieval metrics"""
    model.eval()
    
    all_vision = []
    all_language = []
    all_labels = []
    all_species_logits = []
    all_genus_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            labels = batch['species_idx'].to(device)
            
            outputs = model(vision_emb, language_emb)
            
            all_vision.append(outputs['vision_universal_raw'].cpu())
            all_language.append(outputs['language_universal_raw'].cpu())
            all_labels.append(labels.cpu())
            all_species_logits.append(outputs['species_logits'].cpu())
            
            # Get genus labels for this batch
            genus_labels = []
            for species_name in batch['species']:
                genus = species_to_genus.get(species_name, 'Unknown')
                genus_idx = genus_mapping.get(genus, 0)
                genus_labels.append(genus_idx)
            all_genus_labels.extend(genus_labels)
    
    # Concatenate
    all_vision = torch.cat(all_vision)
    all_language = torch.cat(all_language)
    all_labels = torch.cat(all_labels)
    all_species_logits = torch.cat(all_species_logits)
    all_genus_labels = torch.tensor(all_genus_labels)
    
    # Retrieval metrics
    vision_norm = F.normalize(all_vision, dim=1)
    language_norm = F.normalize(all_language, dim=1)
    similarities = torch.matmul(vision_norm, language_norm.T)
    
    n = len(all_vision)
    metrics = {
        'v2l_r1': 0, 'v2l_r5': 0, 'l2v_r1': 0, 'l2v_r5': 0,
        'v2l_genus_r1': 0, 'v2l_genus_r5': 0,
        'l2v_genus_r1': 0, 'l2v_genus_r5': 0
    }
    
    for i in range(n):
        # V→L retrieval
        _, indices = similarities[i].topk(min(5, n))
        retrieved_labels = all_labels[indices]
        retrieved_genus = all_genus_labels[indices]
        
        # Species-level
        if all_labels[i] in retrieved_labels[:1]:
            metrics['v2l_r1'] += 1
        if all_labels[i] in retrieved_labels[:5]:
            metrics['v2l_r5'] += 1
            
        # Genus-level
        if all_genus_labels[i] in retrieved_genus[:1]:
            metrics['v2l_genus_r1'] += 1
        if all_genus_labels[i] in retrieved_genus[:5]:
            metrics['v2l_genus_r5'] += 1
            
        # L→V retrieval
        _, indices = similarities[:, i].topk(min(5, n))
        retrieved_labels = all_labels[indices]
        retrieved_genus = all_genus_labels[indices]
        
        # Species-level
        if all_labels[i] in retrieved_labels[:1]:
            metrics['l2v_r1'] += 1
        if all_labels[i] in retrieved_labels[:5]:
            metrics['l2v_r5'] += 1
            
        # Genus-level
        if all_genus_labels[i] in retrieved_genus[:1]:
            metrics['l2v_genus_r1'] += 1
        if all_genus_labels[i] in retrieved_genus[:5]:
            metrics['l2v_genus_r5'] += 1
    
    # Normalize retrieval metrics
    for key in metrics:
        metrics[key] /= n
    
    # Classification accuracy
    species_pred = all_species_logits.argmax(dim=1)
    metrics['species_acc'] = (species_pred == all_labels).float().mean().item()
    
    # Top-5 species accuracy
    _, top5_pred = all_species_logits.topk(min(5, all_species_logits.size(1)), dim=1)
    top5_correct = top5_pred.eq(all_labels.view(-1, 1).expand_as(top5_pred))
    metrics['species_acc_top5'] = top5_correct.any(dim=1).float().mean().item()
    
    # Genus-level accuracy from species predictions
    # Map predicted species indices to genus indices
    pred_genus = []
    for pred_idx in species_pred:
        # This is approximate - we'd need the reverse mapping
        pred_genus.append(all_genus_labels[0])  # Placeholder
    
    # For now, compute genus accuracy by checking if any of top-5 species predictions
    # belong to the correct genus (this would need the full species list)
    
    # Debug info
    metrics['unique_preds'] = len(torch.unique(species_pred))
    metrics['unique_labels'] = len(torch.unique(all_labels))
    
    # Add computed averages
    metrics['avg_r1'] = (metrics['v2l_r1'] + metrics['l2v_r1']) / 2
    metrics['avg_r5'] = (metrics['v2l_r5'] + metrics['l2v_r5']) / 2
    metrics['avg_genus_r1'] = (metrics['v2l_genus_r1'] + metrics['l2v_genus_r1']) / 2
    metrics['avg_genus_r5'] = (metrics['v2l_genus_r5'] + metrics['l2v_genus_r5']) / 2
    
    return metrics


def extract_genus_family(species_name):
    """Extract genus and family from species name"""
    # Species names are typically in format "Genus species"
    parts = species_name.split()
    genus = parts[0] if parts else "Unknown"
    
    # For this dataset, we don't have family info directly, 
    # so we'll use genus as a proxy for higher-level grouping
    return genus


def create_hierarchical_mappings(species_mapping):
    """Create genus and family mappings from species"""
    genus_to_species = {}
    species_to_genus = {}
    
    for species, idx in species_mapping.items():
        genus = extract_genus_family(species)
        species_to_genus[species] = genus
        
        if genus not in genus_to_species:
            genus_to_species[genus] = []
        genus_to_species[genus].append(species)
    
    # Create genus mapping
    genus_list = sorted(genus_to_species.keys())
    genus_mapping = {genus: idx for idx, genus in enumerate(genus_list)}
    
    print(f"\nHierarchical taxonomy:")
    print(f"  Total genera: {len(genus_mapping)}")
    print(f"  Average species per genus: {len(species_mapping) / len(genus_mapping):.1f}")
    
    # Show top genera
    genus_counts = [(g, len(species)) for g, species in genus_to_species.items()]
    genus_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 genera by species count:")
    for genus, count in genus_counts[:5]:
        print(f"  {genus}: {count} species")
    
    return genus_mapping, species_to_genus, genus_to_species


def load_train_test_split(config_path):
    """Load train/test split from configuration file with observation mappings"""
    import json
    
    print(f"Loading train/test split from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    train_ids = []
    test_ids = []
    test_reasons = {'temporal_test': 0, 'spatial_test': 0}
    
    for obs_id, metadata in observation_mappings.items():
        if metadata['split'] == 'train':
            train_ids.append(obs_id)
        elif metadata['split'] == 'test':
            test_ids.append(obs_id)
            # Track test split reasons
            reason = metadata.get('split_reason', 'unknown')
            if reason in test_reasons:
                test_reasons[reason] += 1
    
    print(f"Loaded split: {len(train_ids)} train, {len(test_ids)} test observations")
    print(f"Test split reasons: {test_reasons}")
    
    return train_ids, test_ids, observation_mappings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--detach-epochs', type=int, default=3)
    parser.add_argument('--alpha-species-max', type=float, default=0.5)
    parser.add_argument('--eval-every', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-dir', type=str, default='checkpoints_balanced')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--config', type=str, default='../training/config/central_florida_split.json',
                        help='Path to train/test split configuration')
    parser.add_argument('--augment-test-set', action='store_true',
                        help='Whether to augment small test set with more samples')
    args = parser.parse_args()
    
    set_all_seeds(42)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("🌍 Balanced MLP U-Net Training")
    print(f"Device: {device}")
    
    # Load data
    print("\n📊 Loading data...")
    os.chdir(dashboard_path)
    cache = UnifiedDataCache("dataset_config.json")
    
    # Load pre-defined splits
    train_indices, test_indices = load_train_test_split(args.splits_dir, cache)
    
    # Create more balanced splits from the indices
    train_ids, test_ids, valid_species = create_balanced_splits_from_indices(
        train_indices, test_indices, cache,
        min_samples_per_species=args.min_samples_per_species,
        max_test_ratio=args.max_test_ratio
    )
    
    # Create species mapping
    valid_species_sorted = sorted(list(valid_species))
    species_mapping = {species: idx for idx, species in enumerate(valid_species_sorted)}
    
    print(f"\nFinal split statistics:")
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Test IDs: {len(test_ids)}")
    print(f"  Test ratio: {len(test_ids) / (len(train_ids) + len(test_ids)):.2%}")
    # Create hierarchical mappings
    genus_mapping, species_to_genus, genus_to_species = create_hierarchical_mappings(species_mapping)
    
    # Create datasets with the SAME species mapping
    train_dataset = DeepEarthMLPDataset(
        train_ids, cache, 'both', 'cpu', 
        species_mapping=species_mapping,
        augment=True
    )
    test_dataset = DeepEarthMLPDataset(
        test_ids, cache, 'both', 'cpu', 
        species_mapping=species_mapping,
        augment=False
    )
    
    print(f"\nDataset stats:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Species classes: {train_dataset.num_classes}")
    
    # Verify species distribution
    train_species_labels = train_dataset.species_labels.cpu().numpy()
    test_species_labels = test_dataset.species_labels.cpu().numpy()
    print(f"  Unique species in train: {len(np.unique(train_species_labels))}")
    print(f"  Unique species in test: {len(np.unique(test_species_labels))}")
    print(f"  Train label range: {train_species_labels.min()}-{train_species_labels.max()}")
    print(f"  Test label range: {test_species_labels.min()}-{test_species_labels.max()}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = BalancedMLPUNet(
        universal_dim=2048,
        projection_dim=256,
        dropout=0.3,
        num_classes=train_dataset.num_classes
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Species head output dimension: {model.species_head[-1].out_features}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Memory bank
    memory_bank = MemoryBank(size=2048, dim=256)
    
    # Training
    print("\n🚀 Starting training...")
    best_combined = 0
    
    for epoch in range(args.epochs):
        model.train()
        detach_species = (epoch < args.detach_epochs)
        
        epoch_losses = {}
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, batch in enumerate(progress):
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            labels = batch['species_idx'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(vision_emb, language_emb, detach_species=detach_species)
            
            # Debug on first batch of first epoch
            debug_this_batch = args.debug and (epoch == 0 and batch_idx == 0)
            
            losses = compute_balanced_losses(
                outputs, labels, epoch, 
                alpha_species_max=args.alpha_species_max,
                memory_bank=memory_bank,
                debug=debug_this_batch
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update memory bank
            with torch.no_grad():
                memory_bank.update(
                    outputs['vision_universal'].detach(),
                    outputs['language_universal'].detach(),
                    labels
                )
            
            # Track
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item() if torch.is_tensor(v) else v
            
            progress.set_postfix({
                'loss': f"{losses['total'].item():.3f}",
                'sp_acc': f"{losses['species_acc']:.2f}",
                'sp_top5': f"{losses['species_acc_top5']:.2f}",
                'α_sp': f"{losses['alpha_species']:.2f}"
            })
        
        # Average
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        
        print(f"\nEpoch {epoch+1}: Loss={epoch_losses['total']:.3f}, "
              f"Species Top-1={epoch_losses['species_acc']:.2%}, "
              f"Species Top-5={epoch_losses.get('species_acc_top5', 0):.2%}, "
              f"Cos_sim={epoch_losses['cos_sim']:.3f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            results = evaluate_model_with_genus(model, test_loader, device, 
                                              species_to_genus, genus_mapping)
            print(f"\n📊 Evaluation:")
            print(f"  Species-level:")
            print(f"    Retrieval R@1: {results['avg_r1']:.1%} "
                  f"(V→L: {results['v2l_r1']:.1%}, L→V: {results['l2v_r1']:.1%})")
            print(f"    Retrieval R@5: {results['avg_r5']:.1%} "
                  f"(V→L: {results['v2l_r5']:.1%}, L→V: {results['l2v_r5']:.1%})")
            print(f"    Classification Top-1: {results['species_acc']:.1%}")
            print(f"    Classification Top-5: {results['species_acc_top5']:.1%}")
            
            print(f"  Genus-level:")
            print(f"    Retrieval R@1: {results['avg_genus_r1']:.1%} "
                  f"(V→L: {results['v2l_genus_r1']:.1%}, L→V: {results['l2v_genus_r1']:.1%})")
            print(f"    Retrieval R@5: {results['avg_genus_r5']:.1%} "
                  f"(V→L: {results['v2l_genus_r5']:.1%}, L→V: {results['l2v_genus_r5']:.1%})")
            
            print(f"  Diversity: {results['unique_preds']}/{results['unique_labels']} unique predictions")
            
            # Save best
            combined_score = results['avg_r1'] + results['species_acc']
            if combined_score > best_combined:
                best_combined = combined_score
                save_path = Path(args.save_dir) / 'best_model.pth'
                save_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'results': results,
                    'species_mapping': species_mapping
                }, save_path)
                print(f"  💾 New best model! Combined: {combined_score:.1%}")
    
    print("\n✅ Training complete!")
    print(f"Best combined score: {best_combined:.1%}")


if __name__ == "__main__":
    main()
