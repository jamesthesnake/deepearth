#!/usr/bin/env python3
"""
Sophisticated Masking Strategies for DeepEarth Cross-Modal Training
Implements various masking approaches for multimodal learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
from typing import Dict, Tuple, Optional, List
import random


class MaskingStrategy(Enum):
    """Different masking strategies for training"""
    FULL_MODALITY = "full_modality"  # Mask entire modality
    PARTIAL_RANDOM = "partial_random"  # Random partial masking
    BLOCK_MASKING = "block_masking"  # Contiguous blocks
    PROGRESSIVE = "progressive"  # Gradually increase masking
    COMPLEMENTARY = "complementary"  # Mask different parts of different modalities
    SPATIAL_AWARE = "spatial_aware"  # For vision: mask spatial regions
    SEMANTIC_AWARE = "semantic_aware"  # Mask based on importance


class SophisticatedMasking:
    """Advanced masking strategies for multimodal training"""
    
    def __init__(self, 
                 base_mask_prob: float = 0.5,
                 min_mask_prob: float = 0.1,
                 max_mask_prob: float = 0.9,
                 block_size_range: Tuple[int, int] = (1, 5),
                 spatial_block_size: Tuple[int, int] = (4, 8)):
        
        self.base_mask_prob = base_mask_prob
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
        self.block_size_range = block_size_range
        self.spatial_block_size = spatial_block_size
        
    def get_mask_probability(self, epoch: int, max_epochs: int, strategy: str = "cosine") -> float:
        """Dynamic mask probability based on training progress"""
        progress = epoch / max_epochs
        
        if strategy == "linear":
            # Linearly increase masking difficulty
            return self.min_mask_prob + (self.max_mask_prob - self.min_mask_prob) * progress
        
        elif strategy == "cosine":
            # Cosine annealing for smooth progression
            cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            return self.min_mask_prob + (self.max_mask_prob - self.min_mask_prob) * (1 - cosine_factor)
        
        elif strategy == "step":
            # Step-wise increase
            if progress < 0.3:
                return self.min_mask_prob
            elif progress < 0.7:
                return self.base_mask_prob
            else:
                return self.max_mask_prob
        
        else:
            return self.base_mask_prob
    
    def partial_random_mask(self, 
                           embeddings: torch.Tensor, 
                           mask_prob: float,
                           min_keep: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random masking with minimum kept tokens"""
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Ensure we keep at least min_keep fraction
        effective_mask_prob = min(mask_prob, 1.0 - min_keep)
        
        # Generate random mask
        mask = torch.rand(batch_size, device=device) < effective_mask_prob
        
        # Ensure at least one sample is masked (for loss calculation)
        if not mask.any():
            mask[torch.randint(batch_size, (1,))] = True
            
        return mask, embeddings
    
    def block_mask(self, 
                   embeddings: torch.Tensor,
                   mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask contiguous blocks of samples"""
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Calculate number of samples to mask
        num_to_mask = int(batch_size * mask_prob)
        
        if num_to_mask > 0:
            # Random block size
            block_size = random.randint(*self.block_size_range)
            block_size = min(block_size, num_to_mask)
            
            # Random starting position
            max_start = batch_size - block_size
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                mask[start_idx:start_idx + block_size] = True
        
        return mask, embeddings
    
    def spatial_mask_vision(self, 
                           vision_embeddings: torch.Tensor,
                           mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spatial-aware masking for vision embeddings (B, T, H, W, C)"""
        B, T, H, W, C = vision_embeddings.shape
        device = vision_embeddings.device
        
        # Create spatial mask
        masked_vision = vision_embeddings.clone()
        batch_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        for b in range(B):
            if torch.rand(1).item() < mask_prob:
                batch_mask[b] = True
                
                # Random spatial block size
                h_block = random.randint(*self.spatial_block_size)
                w_block = random.randint(*self.spatial_block_size)
                
                # Ensure block fits
                h_block = min(h_block, H)
                w_block = min(w_block, W)
                
                # Random position
                h_start = random.randint(0, H - h_block)
                w_start = random.randint(0, W - w_block)
                
                # Apply spatial mask across all time steps
                masked_vision[b, :, h_start:h_start+h_block, w_start:w_start+w_block, :] = 0
        
        return batch_mask, masked_vision
    
    def complementary_mask(self,
                          vision_emb: torch.Tensor,
                          language_emb: torch.Tensor,
                          total_mask_prob: float) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Complementary masking - when one modality is heavily masked, the other is lightly masked"""
        batch_size = vision_emb.shape[0]
        device = vision_emb.device
        
        # Randomly decide primary masked modality for each sample
        primary_vision = torch.rand(batch_size, device=device) < 0.5
        
        vision_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        language_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            if primary_vision[i]:
                # Heavy masking on vision, light on language
                vision_mask[i] = torch.rand(1, device=device) < total_mask_prob
                language_mask[i] = torch.rand(1, device=device) < (total_mask_prob * 0.2)
            else:
                # Heavy masking on language, light on vision
                language_mask[i] = torch.rand(1, device=device) < total_mask_prob
                vision_mask[i] = torch.rand(1, device=device) < (total_mask_prob * 0.2)
        
        return {
            'vision': (vision_mask, vision_emb),
            'language': (language_mask, language_emb)
        }
    
    def importance_weighted_mask(self,
                                embeddings: torch.Tensor,
                                importance_scores: torch.Tensor,
                                mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask based on importance scores (e.g., attention weights)"""
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Normalize importance scores
        importance_probs = F.softmax(importance_scores, dim=0)
        
        # Invert scores - less important = higher mask probability
        mask_weights = 1.0 - importance_probs
        mask_weights = mask_weights / mask_weights.sum() * batch_size * mask_prob
        
        # Sample based on weights
        mask = torch.rand(batch_size, device=device) < mask_weights
        
        return mask, embeddings


class MultiModalMaskedAutoencoder(nn.Module):
    """Enhanced autoencoder with sophisticated masking strategies"""
    
    def __init__(self, 
                 vision_dim=1408,
                 language_dim=7168,
                 universal_dim=2048,
                 hidden_dims=[512, 256],
                 masking_config=None):
        super().__init__()
        
        # Initialize masking strategies
        self.masking = SophisticatedMasking(**(masking_config or {}))
        
        # Vision pathway
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1)
        )
        self.vision_to_universal = nn.Linear(hidden_dims[1], universal_dim)
        
        # Language pathway
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1)
        )
        self.language_to_universal = nn.Linear(hidden_dims[1], universal_dim)
        
        # Decoders
        self.vision_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], vision_dim)
        )
        
        self.language_decoder = nn.Sequential(
            nn.Linear(universal_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], language_dim)
        )
        
        # Importance scorer (for semantic-aware masking)
        self.importance_scorer = nn.Sequential(
            nn.Linear(universal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, 
                vision_emb: torch.Tensor,
                language_emb: torch.Tensor,
                masking_strategy: MaskingStrategy = MaskingStrategy.PARTIAL_RANDOM,
                mask_prob: float = 0.5,
                epoch: int = 0,
                max_epochs: int = 100):
        
        batch_size = vision_emb.shape[0]
        device = vision_emb.device
        
        # Pool vision embeddings
        vision_pooled = vision_emb.mean(dim=(1, 2, 3))
        
        # Encode to universal space
        vision_hidden = self.vision_encoder(vision_pooled)
        vision_universal = self.vision_to_universal(vision_hidden)
        
        language_hidden = self.language_encoder(language_emb)
        language_universal = self.language_to_universal(language_hidden)
        
        # Calculate importance scores for semantic masking
        vision_importance = self.importance_scorer(vision_universal).squeeze()
        language_importance = self.importance_scorer(language_universal).squeeze()
        
        # Apply masking strategy
        if masking_strategy == MaskingStrategy.FULL_MODALITY:
            # Standard approach - mask one full modality
            if epoch % 2 == 0:
                vision_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                language_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                vision_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                language_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            masked_vision = vision_pooled
            masked_language = language_emb
            
        elif masking_strategy == MaskingStrategy.PARTIAL_RANDOM:
            # Dynamic probability based on training progress
            dynamic_prob = self.masking.get_mask_probability(epoch, max_epochs, "cosine")
            vision_mask, masked_vision = self.masking.partial_random_mask(vision_pooled, dynamic_prob)
            language_mask, masked_language = self.masking.partial_random_mask(language_emb, dynamic_prob)
            
        elif masking_strategy == MaskingStrategy.BLOCK_MASKING:
            vision_mask, masked_vision = self.masking.block_mask(vision_pooled, mask_prob)
            language_mask, masked_language = self.masking.block_mask(language_emb, mask_prob)
            
        elif masking_strategy == MaskingStrategy.COMPLEMENTARY:
            mask_results = self.masking.complementary_mask(vision_pooled, language_emb, mask_prob)
            vision_mask, masked_vision = mask_results['vision']
            language_mask, masked_language = mask_results['language']
            
        elif masking_strategy == MaskingStrategy.SPATIAL_AWARE:
            # Special handling for vision spatial masking
            vision_mask, masked_vision_full = self.masking.spatial_mask_vision(vision_emb, mask_prob)
            masked_vision = masked_vision_full.mean(dim=(1, 2, 3))
            language_mask, masked_language = self.masking.partial_random_mask(language_emb, mask_prob)
            
        elif masking_strategy == MaskingStrategy.SEMANTIC_AWARE:
            # Use importance scores for masking
            vision_mask, masked_vision = self.masking.importance_weighted_mask(
                vision_pooled, vision_importance, mask_prob)
            language_mask, masked_language = self.masking.importance_weighted_mask(
                language_emb, language_importance, mask_prob)
        
        else:
            # Progressive masking
            dynamic_prob = self.masking.get_mask_probability(epoch, max_epochs, "linear")
            vision_mask, masked_vision = self.masking.partial_random_mask(vision_pooled, dynamic_prob)
            language_mask, masked_language = self.masking.partial_random_mask(language_emb, dynamic_prob)
        
        # Reconstruction paths with cross-modal information
        # When vision is masked, use language to reconstruct
        vision_reconstructed = self.vision_decoder(language_universal)
        # When language is masked, use vision to reconstruct  
        language_reconstructed = self.language_decoder(vision_universal)
        
        # Calculate losses only on masked portions
        vision_recon_loss = F.mse_loss(
            vision_reconstructed[vision_mask],
            vision_pooled[vision_mask]
        ) if vision_mask.any() else torch.tensor(0.0, device=device)
        
        language_recon_loss = F.mse_loss(
            language_reconstructed[language_mask],
            language_emb[language_mask]
        ) if language_mask.any() else torch.tensor(0.0, device=device)
        
        # Alignment loss (always computed)
        alignment_loss = 1 - F.cosine_similarity(vision_universal, language_universal).mean()
        
        # Diversity loss to prevent collapse
        vision_std = vision_universal.std(dim=0).mean()
        language_std = language_universal.std(dim=0).mean()
        diversity_loss = -torch.log(vision_std + 1e-6) - torch.log(language_std + 1e-6)
        
        # Combined loss with adaptive weighting
        total_loss = (vision_recon_loss + language_recon_loss) + \
                    0.5 * alignment_loss + \
                    0.1 * diversity_loss
        
        return {
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'vision_reconstructed': vision_reconstructed,
            'language_reconstructed': language_reconstructed,
            'vision_mask': vision_mask,
            'language_mask': language_mask,
            'vision_recon_loss': vision_recon_loss,
            'language_recon_loss': language_recon_loss,
            'alignment_loss': alignment_loss,
            'diversity_loss': diversity_loss,
            'total_loss': total_loss,
            'mask_stats': {
                'vision_masked_ratio': vision_mask.float().mean().item(),
                'language_masked_ratio': language_mask.float().mean().item(),
                'vision_importance_mean': vision_importance.mean().item(),
                'language_importance_mean': language_importance.mean().item()
            }
        }


def train_with_sophisticated_masking(model, train_loader, test_loader, 
                                   epochs=30, device='cuda',
                                   masking_schedule=None):
    """Training loop with sophisticated masking strategies"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Default masking schedule
    if masking_schedule is None:
        masking_schedule = {
            range(0, 5): MaskingStrategy.PARTIAL_RANDOM,      # Warm up
            range(5, 10): MaskingStrategy.BLOCK_MASKING,     # Learn structure
            range(10, 15): MaskingStrategy.COMPLEMENTARY,    # Cross-modal
            range(15, 20): MaskingStrategy.SPATIAL_AWARE,    # Spatial understanding
            range(20, 25): MaskingStrategy.SEMANTIC_AWARE,   # Importance-based
            range(25, 30): MaskingStrategy.PROGRESSIVE       # Final push
        }
    
    metrics = {
        'losses': [],
        'mask_strategies': [],
        'mask_ratios': [],
        'alignment_scores': []
    }
    
    for epoch in range(epochs):
        # Determine current masking strategy
        current_strategy = MaskingStrategy.PARTIAL_RANDOM
        for epoch_range, strategy in masking_schedule.items():
            if epoch in epoch_range:
                current_strategy = strategy
                break
        
        print(f"\nEpoch {epoch+1}/{epochs} - Strategy: {current_strategy.value}")
        
        model.train()
        epoch_metrics = {
            'total_loss': 0,
            'vision_mask_ratio': 0,
            'language_mask_ratio': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            
            # Forward pass with current masking strategy
            outputs = model(
                vision, language,
                masking_strategy=current_strategy,
                mask_prob=0.5,
                epoch=epoch,
                max_epochs=epochs
            )
            
            loss = outputs['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['vision_mask_ratio'] += outputs['mask_stats']['vision_masked_ratio']
            epoch_metrics['language_mask_ratio'] += outputs['mask_stats']['language_masked_ratio']
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"V_mask={outputs['mask_stats']['vision_masked_ratio']:.2f}, "
                      f"L_mask={outputs['mask_stats']['language_masked_ratio']:.2f}")
        
        # Average metrics
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        metrics['losses'].append(epoch_metrics['total_loss'])
        metrics['mask_strategies'].append(current_strategy.value)
        metrics['mask_ratios'].append({
            'vision': epoch_metrics['vision_mask_ratio'],
            'language': epoch_metrics['language_mask_ratio']
        })
        
        scheduler.step()
        
        # Evaluation
        if (epoch + 1) % 5 == 0:
            model.eval()
            alignments = []
            
            with torch.no_grad():
                for batch in test_loader:
                    vision = batch['vision_embedding'].to(device)
                    language = batch['language_embedding'].to(device)
                    
                    outputs = model(
                        vision, language,
                        masking_strategy=MaskingStrategy.PARTIAL_RANDOM,
                        mask_prob=0.0  # No masking during evaluation
                    )
                    
                    cos_sim = F.cosine_similarity(
                        outputs['vision_universal'],
                        outputs['language_universal']
                    )
                    alignments.extend(cos_sim.cpu().numpy())
            
            alignment_score = np.mean(alignments)
            metrics['alignment_scores'].append(alignment_score)
            
            print(f"\nEvaluation - Alignment: {alignment_score:.3f}")
    
    return metrics


# Example usage and visualization
if __name__ == "__main__":
    print("🎭 Sophisticated Masking Strategies Demo\n")
    
    # Create dummy data
    batch_size = 32
    vision_emb = torch.randn(batch_size, 8, 24, 24, 1408)
    language_emb = torch.randn(batch_size, 7168)
    
    # Initialize model with masking config
    masking_config = {
        'base_mask_prob': 0.5,
        'min_mask_prob': 0.1,
        'max_mask_prob': 0.8,
        'block_size_range': (2, 6),
        'spatial_block_size': (6, 6)
    }
    
    model = MultiModalMaskedAutoencoder(masking_config=masking_config)
    
    # Test different masking strategies
    strategies = [
        MaskingStrategy.FULL_MODALITY,
        MaskingStrategy.PARTIAL_RANDOM,
        MaskingStrategy.BLOCK_MASKING,
        MaskingStrategy.COMPLEMENTARY,
        MaskingStrategy.SPATIAL_AWARE,
        MaskingStrategy.SEMANTIC_AWARE
    ]
    
    print("Testing masking strategies:")
    for strategy in strategies:
        outputs = model(vision_emb, language_emb, masking_strategy=strategy)
        
        print(f"\n{strategy.value}:")
        print(f"  Vision masked: {outputs['mask_stats']['vision_masked_ratio']:.2%}")
        print(f"  Language masked: {outputs['mask_stats']['language_masked_ratio']:.2%}")
        print(f"  Total loss: {outputs['total_loss'].item():.4f}")
