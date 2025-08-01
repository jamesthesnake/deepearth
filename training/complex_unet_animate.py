#!/usr/bin/env python3
"""
Fixed Multimodal Autoencoder with Input Masking and MLP U-Net Architecture
Critical fixes applied:
1. Fixed decoder dimension mismatch when using dimension reduction
2. Fixed LayerNorm dimensions for post-encoder normalization
3. Improved patch encoder with attention-based aggregation
4. Fixed contrastive loss to handle small batches properly
5. Better handling of fusion mechanisms with mixed token/vector inputs
6. Decoder hidden dimensions adjusted for dimension reduction
7. Added helper function for computing decoder dimensions safely
8. Improved weight initialization for large matrices

Latest fixes (v2):
9. Removed duplicate dimension reduction in forward pass
10. Fixed vision/language_latent_high to preserve 2048D representations
11. Improved skip projector bookkeeping in decoder
12. Optimized mixed precision to prefer bfloat16 on H100
13. Added pytorch-metric-learning version check for miner reset

Latest fixes (v3):
14. Fixed contrastive accuracy to use B×B similarity (ignoring mined pairs)
15. Clean latents now exclude type embeddings for pure content alignment
16. Removed dead FP8 code and duplicate checks
17. Improved skip mapping warnings for decoder depth mismatches
18. Better miner reset warnings for PML version requirements

Latest fixes (v4):
19. Fixed warmup scheduler to be step-based (not epoch-based)
20. Fixed decoder interpolation to target universal_dim (not input dim)
21. Added memmap TODO for full-scale dataset
22. Fixed SequentialLR to use step-based milestones
23. Fixed L→V retrieval to use averaged embeddings (not first)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import random
from tqdm import tqdm

# Add dashboard to path
dashboard_path = Path(__file__).parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from services.training_data import get_training_batch, get_available_observation_ids
from data_cache import UnifiedDataCache

# Import improvements
try:
    from multimodal_improvements import (
        ContrastiveLoss, 
        UniversalSpaceRegularizer, 
        analyze_universal_space
    )
    IMPROVEMENTS_AVAILABLE = True
except ImportError:
    logging.warning("multimodal_improvements.py not found. Running without some features.")
    IMPROVEMENTS_AVAILABLE = False

# Try to import pytorch-metric-learning
try:
    from pytorch_metric_learning import losses, miners, distances
    PML_AVAILABLE = True
except ImportError:
    logging.warning("pytorch-metric-learning not found. Using simple contrastive loss.")
    PML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_decoder_hidden_dims(latent_dim: int, encoder_dims: List[int], 
                               width_factor: float = 1.0, use_dimension_reduction: bool = False) -> List[int]:
    """
    Compute decoder hidden dimensions based on encoder dimensions and latent dimension.
    
    Args:
        latent_dim: Dimension of latent space (after reduction if applicable)
        encoder_dims: List of encoder hidden dimensions
        width_factor: Scaling factor for decoder width (default 1.0)
        use_dimension_reduction: Whether dimension reduction is being used
        
    Returns:
        List of decoder hidden dimensions
    """
    # Reverse encoder dims to get natural decoder progression
    reversed_encoder = encoder_dims[::-1]
    
    if use_dimension_reduction:
        # When using dimension reduction, we need to carefully size hidden layers
        decoder_dims = []
        
        # First layer: needs to expand from latent_dim
        first_dim = int(latent_dim * 2 * width_factor)
        decoder_dims.append(first_dim)
        
        # Middle layers: gradual expansion toward the universal dimension (not input dim)
        # Use the first encoder hidden dim (universal_dim) as target
        target_dim = reversed_encoder[0]  # This is the universal_dim
        
        for i in range(1, len(reversed_encoder) - 1):
            # Interpolate between first_dim and universal_dim
            progress = i / (len(reversed_encoder) - 1)
            interpolated = int(first_dim + (target_dim - first_dim) * progress)
            # Ensure monotonic increase
            current_dim = max(interpolated, decoder_dims[-1])
            decoder_dims.append(int(current_dim * width_factor))
        
        # Final layer: match original input dimension
        decoder_dims.append(reversed_encoder[-1])
    else:
        # Without dimension reduction, mirror encoder architecture
        decoder_dims = [int(d * width_factor) for d in reversed_encoder[:-1]]
        decoder_dims.append(reversed_encoder[-1])  # Final dim unchanged
    
    return decoder_dims


def init_large_layer(module: nn.Module, fan_in_threshold: int = 2048):
    """
    Initialize large layers with truncated normal for better convergence.
    
    Args:
        module: PyTorch module to initialize
        fan_in_threshold: Threshold for considering a layer "large"
    """
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        if fan_in > fan_in_threshold:
            # Use truncated normal for large layers
            nn.init.trunc_normal_(module.weight, std=0.01)
            logger.debug(f"Initialized large layer ({fan_in} inputs) with truncated normal")
        else:
            # Use Xavier for smaller layers
            nn.init.xavier_normal_(module.weight, gain=0.02)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# FIXED: Vision Patch Encoder with proper aggregation
class VisionPatchEncoder(nn.Module):
    """
    Extract patch tokens from vision embeddings
    SIMPLIFIED: Since we're not using tokens downstream, just do efficient pooling
    """
    def __init__(self, input_dim=1408, patch_size=4, output_dim=512, num_patches=36,
                 use_attention_pool=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.use_attention_pool = use_attention_pool
        
        if use_attention_pool:
            # Attention-based pooling over spatial patches
            self.patch_proj = nn.Linear(input_dim, output_dim)
            # Learn to weight different spatial regions
            self.spatial_attention = nn.Sequential(
                nn.Linear(output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.norm = nn.LayerNorm(output_dim)
        else:
            # Simple projection after mean pooling
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
    def forward(self, x):
        # x: (B, 8, 24, 24, 1408) or (B, 1408) if already mean-pooled
        if x.dim() == 2:
            # Already mean-pooled, just project
            if self.use_attention_pool:
                return self.norm(self.patch_proj(x))
            else:
                return self.proj(x)
            
        B, T, H, W, C = x.shape
        
        # Take middle temporal frame
        x = x[:, T//2]  # (B, 24, 24, 1408)
        
        if self.use_attention_pool:
            # Extract patches using adaptive pooling
            x = x.permute(0, 3, 1, 2)  # (B, 1408, 24, 24)
            x = F.adaptive_avg_pool2d(x, (6, 6))  # (B, 1408, 6, 6)
            x = x.permute(0, 2, 3, 1).reshape(B, 36, C)  # (B, 36, 1408)
            
            # Project patches
            patches = self.patch_proj(x)  # (B, 36, output_dim)
            
            # Compute attention weights
            attn_logits = self.spatial_attention(patches)  # (B, 36, 1)
            attn_weights = F.softmax(attn_logits, dim=1)
            
            # Weighted aggregation
            aggregated = (patches * attn_weights).sum(dim=1)  # (B, output_dim)
            return self.norm(aggregated)
        else:
            # Simple mean pooling + projection
            x = x.mean(dim=(1, 2))  # (B, 1408)
            return self.proj(x)


# Dimension Distiller remains the same
class DimensionDistiller(nn.Module):
    """Distill from high-dim to low-dim representations"""
    def __init__(self, teacher_dim=2048, student_dim=512, temperature=4.0):
        super().__init__()
        self.projector = nn.Linear(teacher_dim, student_dim)
        self.temperature = temperature
        
    def forward(self, teacher_features, return_loss=False):
        student_features = self.projector(teacher_features)
        
        if return_loss:
            # Cosine similarity distillation
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            student_norm = F.normalize(student_features, p=2, dim=-1)
            
            # Handle both single vectors and multiple tokens
            if teacher_norm.dim() == 2:
                # Self-similarity matrices
                teacher_sim = torch.mm(teacher_norm, teacher_norm.t()) / self.temperature
                student_sim = torch.mm(student_norm, student_norm.t()) / self.temperature
            else:
                # Token-wise similarity
                teacher_sim = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2)) / self.temperature
                student_sim = torch.bmm(student_norm, student_norm.transpose(1, 2)) / self.temperature
            
            # KL divergence between similarity distributions
            loss = F.kl_div(
                F.log_softmax(student_sim.view(-1, student_sim.shape[-1]), dim=-1),
                F.softmax(teacher_sim.view(-1, teacher_sim.shape[-1]), dim=-1),
                reduction='batchmean'
            )
            return student_features, loss
        
        return student_features


# Fusion Registry
class FusionRegistry:
    """Registry for different fusion mechanisms"""
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def decorator(fusion_class):
            cls._registry[name] = fusion_class
            return fusion_class
        return decorator
    
    @classmethod
    def get(cls, name):
        return cls._registry.get(name)
    
    @classmethod
    def list(cls):
        return list(cls._registry.keys())


@FusionRegistry.register('mlp')
class MLPFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, vision, language):
        # Handle token dimension
        if vision.dim() == 3 and language.dim() == 2:
            vision = vision.mean(dim=1)
        elif vision.dim() == 2 and language.dim() == 3:
            language = language.mean(dim=1)
        elif vision.dim() == 3 and language.dim() == 3:
            vision = vision.mean(dim=1)
            language = language.mean(dim=1)
            
        fused = self.fusion(torch.cat([vision, language], dim=-1))
        return fused, fused


# FIXED: Better token handling in cross-attention
@FusionRegistry.register('multi_head_attn')
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn_v2l = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_l2v = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        
    def forward(self, vision, language):
        # Track original dimensions
        v_is_single = vision.dim() == 2
        l_is_single = language.dim() == 2
        
        # Ensure both have token dimension
        if v_is_single:
            vision = vision.unsqueeze(1)  # (B, 1, D)
        if l_is_single:
            language = language.unsqueeze(1)  # (B, 1, D)
            
        # Cross attention with residual
        vision_attended, _ = self.attn_v2l(vision, language, language)
        vision_out = self.norm_v(vision + vision_attended)
        
        language_attended, _ = self.attn_l2v(language, vision, vision)
        language_out = self.norm_l(language + language_attended)
        
        # Return in original format
        if v_is_single:
            vision_out = vision_out.squeeze(1)
        if l_is_single:
            language_out = language_out.squeeze(1)
            
        return vision_out, language_out


@FusionRegistry.register('gated_add')
class GatedAdditiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_v = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.gate_l = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, vision, language):
        # Handle token dimension
        if vision.dim() == 3 and language.dim() == 2:
            language = language.unsqueeze(1).expand(-1, vision.shape[1], -1)
        elif vision.dim() == 2 and language.dim() == 3:
            vision = vision.unsqueeze(1).expand(-1, language.shape[1], -1)
        elif vision.dim() == 3 and language.dim() == 3 and vision.shape[1] != language.shape[1]:
            vision = vision.mean(dim=1)
            language = language.mean(dim=1)
            
        concat = torch.cat([vision, language], dim=-1)
        gate_v = self.gate_v(concat)
        gate_l = self.gate_l(concat)
        
        vision_fused = vision + gate_v * language
        language_fused = language + gate_l * vision
        
        return vision_fused, language_fused


@FusionRegistry.register('simple')
class SimpleFusion(nn.Module):
    """Original simple fusion for compatibility"""
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),  # [v, l, |v-l|, v*l]
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, vision, language):
        # Ensure single vectors
        if vision.dim() == 3:
            vision = vision.mean(dim=1)
        if language.dim() == 3:
            language = language.mean(dim=1)
            
        # Compute interaction features
        diff = torch.abs(vision - language)
        prod = vision * language
        
        # Concatenate all features
        combined = torch.cat([vision, language, diff, prod], dim=-1)
        
        # Fuse and add residual
        fused = self.fusion(combined)
        return fused + vision, fused + language


# FIXED: Contrastive loss with better batch size handling
class RobustContrastiveLoss(nn.Module):
    """Contrastive loss using pytorch-metric-learning if available"""
    def __init__(self, temperature=0.07, use_miner=True, min_batch_for_mining=128):
        super().__init__()
        self.temperature = temperature
        self.min_batch_for_mining = min_batch_for_mining
        
        if PML_AVAILABLE:
            self.distance = distances.CosineSimilarity()
            self.loss_func = losses.NTXentLoss(
                temperature=temperature,
                distance=self.distance
            )
            # Only use miner for larger batches
            if use_miner:
                self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
            else:
                self.miner = None
        else:
            self.distance = None
            self.loss_func = None
            self.miner = None
        
    def forward(self, vision_embeds, language_embeds):
        # Handle multi-token inputs by pooling
        if vision_embeds.dim() == 3:
            vision_embeds = vision_embeds.mean(dim=1)
        if language_embeds.dim() == 3:
            language_embeds = language_embeds.mean(dim=1)
            
        batch_size = vision_embeds.shape[0]
        
        if PML_AVAILABLE and self.loss_func is not None:
            labels = torch.arange(batch_size, device=vision_embeds.device)
            
            # Concatenate embeddings
            embeddings = torch.cat([vision_embeds, language_embeds], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            
            # Only use miner for larger batches
            if self.miner is not None and batch_size >= self.min_batch_for_mining:
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_func(embeddings, labels, hard_pairs)
            else:
                loss = self.loss_func(embeddings, labels)
            
            # Compute accuracy on original B×B similarity (ignoring mined pairs)
            with torch.no_grad():
                vision_norm = F.normalize(vision_embeds, p=2, dim=1)
                language_norm = F.normalize(language_embeds, p=2, dim=1)
                sim_matrix = torch.mm(vision_norm, language_norm.t())
                
                # Standard R@1 over B×B matrix
                v2l_acc = (sim_matrix.argmax(dim=1) == torch.arange(batch_size, device=sim_matrix.device)).float().mean()
                l2v_acc = (sim_matrix.argmax(dim=0) == torch.arange(batch_size, device=sim_matrix.device)).float().mean()
            
            return loss, {'v2l_acc': v2l_acc.item(), 'l2v_acc': l2v_acc.item()}
        else:
            # Fallback to simple contrastive loss
            return self._simple_contrastive(vision_embeds, language_embeds)
    
    def _simple_contrastive(self, vision_embeds, language_embeds):
        """Simple InfoNCE loss implementation"""
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)
        language_embeds = F.normalize(language_embeds, p=2, dim=1)
        
        logits = torch.mm(vision_embeds, language_embeds.t()) / self.temperature
        labels = torch.arange(len(vision_embeds), device=vision_embeds.device)
        
        loss_v2l = F.cross_entropy(logits, labels)
        loss_l2v = F.cross_entropy(logits.t(), labels)
        loss = (loss_v2l + loss_l2v) / 2
        
        with torch.no_grad():
            v2l_acc = (logits.argmax(dim=1) == labels).float().mean()
            l2v_acc = (logits.t().argmax(dim=1) == labels).float().mean()
        
        return loss, {'v2l_acc': v2l_acc.item(), 'l2v_acc': l2v_acc.item()}


# Simple inline contrastive loss if improvements not available
class SimpleContrastiveLoss(nn.Module):
    """Simple InfoNCE loss for cross-modal alignment"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, vision_embeds, language_embeds):
        # Normalize embeddings
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)
        language_embeds = F.normalize(language_embeds, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(vision_embeds, language_embeds.t()) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(len(vision_embeds), device=vision_embeds.device)
        
        # Compute loss in both directions
        loss_v2l = F.cross_entropy(logits, labels)
        loss_l2v = F.cross_entropy(logits.t(), labels)
        
        # Average the two directions
        loss = (loss_v2l + loss_l2v) / 2
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            v2l_acc = (logits.argmax(dim=1) == labels).float().mean()
            l2v_acc = (logits.t().argmax(dim=1) == labels).float().mean()
        
        return loss, {'v2l_acc': v2l_acc.item(), 'l2v_acc': l2v_acc.item()}


class MLPBlock(nn.Module):
    """Basic MLP block with residual connection"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.net(x) + self.residual(x)


class MLPUNetEncoder(nn.Module):
    """MLP-based U-Net encoder with skip connections"""
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512], dropout=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            self.blocks.append(MLPBlock(dims[i], dims[i+1], dropout))
        
    def forward(self, x):
        skip_connections = []
        
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        
        return x, skip_connections


# FIXED: MLPUNetDecoder with proper dimension handling
class MLPUNetDecoder(nn.Module):
    """
    MLP-based U-Net decoder with skip connections
    FIXED: Handles dimension mismatch when using dimension reduction
    """
    def __init__(self, latent_dim, hidden_dims=[512, 1024, 2048], output_dim=None, 
                 dropout=0.1, skip_mode='concat', encoder_dims=None,
                 dimension_reduced=False, original_latent_dim=None):
        super().__init__()
        
        self.skip_mode = skip_mode
        self.dimension_reduced = dimension_reduced
        self.latent_dim = latent_dim
        self.blocks = nn.ModuleList()
        
        # Store encoder dimensions for proper skip connections
        if encoder_dims is None:
            encoder_dims = hidden_dims[::-1]
        
        # Pre-compute skip mapping to avoid index errors
        self.skip_mapping = self._compute_skip_mapping(len(hidden_dims), len(encoder_dims))
        
        # Create skip projectors if using dimension reduction
        self.skip_projectors = nn.ModuleList()
        
        # Build decoder blocks with correct dimensions
        current_dim = latent_dim
        
        for i in range(len(hidden_dims)):
            # Determine if skip connection is needed and get dimensions
            skip_proj = None
            if i == 0:
                # First block: no skip connection
                in_dim = current_dim
            else:
                # Use pre-computed skip mapping
                skip_idx = self.skip_mapping.get(i, None)
                if skip_idx is not None and 0 <= skip_idx < len(encoder_dims):
                    skip_source_dim = encoder_dims[skip_idx]
                else:
                    # Fallback: use current decoder dimension
                    skip_source_dim = current_dim
                    logger.warning(f"Decoder block {i} has no valid encoder skip, using current_dim={current_dim}")
                
                if skip_mode == 'concat':
                    if dimension_reduced:
                        # Project skip to target dimension (same as latent_dim for consistency)
                        skip_proj = nn.Linear(skip_source_dim, latent_dim)
                        in_dim = current_dim + latent_dim
                    else:
                        # No projection needed
                        in_dim = current_dim + skip_source_dim
                else:  # additive
                    in_dim = current_dim
                    if dimension_reduced and skip_source_dim != current_dim:
                        skip_proj = nn.Linear(skip_source_dim, current_dim)
            
            # Append the projector (None if not needed)
            self.skip_projectors.append(skip_proj)
            
            # Create the decoder block
            out_dim = hidden_dims[i]
            self.blocks.append(MLPBlock(in_dim, out_dim, dropout))
            current_dim = out_dim
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim) if output_dim else None
    
    def _compute_skip_mapping(self, num_decoder_blocks, num_encoder_outputs):
        """Pre-compute deterministic mapping from decoder blocks to encoder skip indices"""
        mapping = {}
        # Block 0 has no skip
        # Block i>0 gets skip from encoder output (num_encoder_outputs - i)
        for i in range(1, num_decoder_blocks):
            encoder_idx = num_encoder_outputs - i
            if 0 <= encoder_idx < num_encoder_outputs:
                mapping[i] = encoder_idx
            else:
                logger.warning(f"Skip mapping out of bounds for decoder block {i} "
                             f"(encoder has {num_encoder_outputs} outputs)")
        return mapping
        
    def forward(self, x, skip_connections, mask=None):
        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1]
        
        for i, block in enumerate(self.blocks):
            if i > 0 and i - 1 < len(skip_connections):
                skip = skip_connections[i - 1]
                
                # Gate skip connections BEFORE projection
                if self.training and mask is not None and skip.shape[0] == mask.shape[0]:
                    skip = skip * (~mask).float().unsqueeze(-1)
                
                # Project skip if we have a projector for this layer
                if i < len(self.skip_projectors) and self.skip_projectors[i] is not None:
                    skip = self.skip_projectors[i](skip)
                    
                if self.skip_mode == 'concat':
                    x = torch.cat([x, skip], dim=-1)
                else:  # additive
                    x = x + skip
                    
            x = block(x)
        
        if self.output_proj:
            x = self.output_proj(x)
        
        return x


class MultimodalMLPUNet(nn.Module):
    """
    FIXED: Multimodal autoencoder with proper dimension handling throughout
    """
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048, 
                 use_vision_patches=True, fusion_type='multi_head_attn',
                 use_dimension_reduction=False, target_dim=512):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        self.use_vision_patches = use_vision_patches
        self.fusion_type = fusion_type
        self.use_dimension_reduction = use_dimension_reduction
        self.target_dim = target_dim
        
        # Vision processing
        if use_vision_patches:
            self.vision_patch_encoder = VisionPatchEncoder(
                input_dim=vision_dim,
                output_dim=vision_dim,  # Keep same dim for U-Net input
                num_patches=36,
                use_attention_pool=True  # Use attention-based spatial pooling
            )
        
        # Vision U-Net
        vision_encoder_dims = [1408, 1792, universal_dim]
        self.vision_encoder = MLPUNetEncoder(
            input_dim=vision_dim,
            hidden_dims=vision_encoder_dims
        )
        
        # Language U-Net
        language_encoder_dims = [5120, 3584, universal_dim]
        self.language_encoder = MLPUNetEncoder(
            input_dim=language_dim,
            hidden_dims=language_encoder_dims
        )
        
        # IMPROVED: Use helper function to compute decoder dimensions
        vision_decoder_hidden = compute_decoder_hidden_dims(
            latent_dim=target_dim if use_dimension_reduction else universal_dim,
            encoder_dims=vision_encoder_dims,
            width_factor=1.0,
            use_dimension_reduction=use_dimension_reduction
        )
        
        language_decoder_hidden = compute_decoder_hidden_dims(
            latent_dim=target_dim if use_dimension_reduction else universal_dim,
            encoder_dims=language_encoder_dims,
            width_factor=1.0,
            use_dimension_reduction=use_dimension_reduction
        )
        
        logger.info(f"Vision decoder dims: {vision_decoder_hidden}")
        logger.info(f"Language decoder dims: {language_decoder_hidden}")
        
        # Create decoders with computed hidden dimensions
        self.vision_decoder = MLPUNetDecoder(
            latent_dim=target_dim if use_dimension_reduction else universal_dim,
            hidden_dims=vision_decoder_hidden,
            output_dim=self.vision_dim,
            skip_mode='concat',
            encoder_dims=vision_encoder_dims,
            dimension_reduced=use_dimension_reduction,
            original_latent_dim=universal_dim
        )
        
        self.language_decoder = MLPUNetDecoder(
            latent_dim=target_dim if use_dimension_reduction else universal_dim,
            hidden_dims=language_decoder_hidden,
            output_dim=self.language_dim,
            skip_mode='concat',
            encoder_dims=language_encoder_dims,
            dimension_reduced=use_dimension_reduction,
            original_latent_dim=universal_dim
        )
        
        # CRITICAL FIX: LayerNorms at correct dimensions
        # These are applied BEFORE dimension reduction, so use universal_dim
        self.post_enc_norm_v = nn.LayerNorm(universal_dim)
        self.post_enc_norm_l = nn.LayerNorm(universal_dim)
        
        # Dimension reduction if requested
        effective_dim = target_dim if use_dimension_reduction else universal_dim
        if use_dimension_reduction:
            self.vision_distiller = DimensionDistiller(universal_dim, target_dim)
            self.language_distiller = DimensionDistiller(universal_dim, target_dim)
            # Add normalization after dimension reduction for stability
            self.post_reduction_norm_v = nn.LayerNorm(target_dim)
            self.post_reduction_norm_l = nn.LayerNorm(target_dim)
        else:
            self.vision_distiller = None
            self.language_distiller = None
            self.post_reduction_norm_v = None
            self.post_reduction_norm_l = None
        
        # Cross-modal fusion
        fusion_class = FusionRegistry.get(fusion_type)
        if fusion_class is None:
            logger.warning(f"Fusion type '{fusion_type}' not found. Using MLP fusion.")
            fusion_class = FusionRegistry.get('mlp')
        self.cross_modal_fusion = fusion_class(effective_dim)
        
        # CRITICAL FIX: Type embeddings at correct dimension (after reduction)
        # Initialize with moderate magnitude (norm ~0.7) and add AFTER LayerNorm
        vision_type_init = torch.randn(effective_dim)
        vision_type_init = F.normalize(vision_type_init, dim=0) * 0.7  # norm = 0.7
        self.vision_type_embedding = nn.Parameter(vision_type_init)
        
        language_type_init = torch.randn(effective_dim)
        language_type_init = F.normalize(language_type_init, dim=0) * 0.7  # norm = 0.7
        self.language_type_embedding = nn.Parameter(language_type_init)
        
        # Learnable mask tokens with MLP projectors for non-trivial input
        self.vision_mask_token = nn.Parameter(torch.randn(vision_dim) * 0.1)
        self.language_mask_token = nn.Parameter(torch.randn(language_dim) * 0.1)
        
        # Mask token projectors
        self.vision_mask_projector = self._create_mask_projector(vision_dim)
        self.language_mask_projector = self._create_mask_projector(language_dim)
        
        # Input normalization
        self.vision_input_norm = nn.LayerNorm(vision_dim)
        self.language_input_norm = nn.LayerNorm(language_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _create_mask_projector(self, dim, hidden_dim=256):
        """Create MLP projector for mask tokens"""
        projector = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        # Initialize with reasonable scale
        for module in projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return projector
        
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use improved initialization for large layers
                init_large_layer(module)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Special initialization for attention weights in patch encoder
        if hasattr(self, 'vision_patch_encoder') and self.use_vision_patches:
            if hasattr(self.vision_patch_encoder, 'spatial_attention'):
                # Initialize final layer to produce near-uniform attention
                final_layer = self.vision_patch_encoder.spatial_attention[-1]
                nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(final_layer.bias)
        
    def forward(self, vision_emb, language_emb, vision_mask_ratio=0.0, language_mask_ratio=1.0, 
                language_loss_weight=1.0, return_distillation_loss=False):
        """Forward pass with proper dimension handling"""
        B = vision_emb.shape[0]
        device = vision_emb.device
        
        # Handle vision embedding shape and patches
        if self.use_vision_patches and vision_emb.dim() == 5:
            # Process through patch encoder
            vision_emb = self.vision_patch_encoder(vision_emb)  # (B, vision_dim)
        elif vision_emb.dim() == 5:
            # Full vision embeddings - mean pool
            vision_emb = vision_emb.mean(dim=(1, 2, 3))  # (B, 1408)
        
        # Normalize inputs
        vision_emb = self.vision_input_norm(vision_emb)
        language_emb = self.language_input_norm(language_emb)
        
        # Store originals for reconstruction loss
        vision_original = vision_emb.clone()
        language_original = language_emb.clone()
        
        # Apply input masking (BEFORE encoding, as per boss's notes)
        if self.training:
            # Vision masking
            vision_mask = torch.rand(B, device=device) < vision_mask_ratio
            if vision_mask.any():
                # Project mask tokens for non-trivial input
                mask_tokens_v = self.vision_mask_projector(
                    self.vision_mask_token.unsqueeze(0).expand(B, -1)
                )
                vision_emb = torch.where(
                    vision_mask.unsqueeze(1),
                    mask_tokens_v,
                    vision_emb
                )
            
            # Language masking
            language_mask = torch.rand(B, device=device) < language_mask_ratio
            if language_mask.any():
                # Project mask tokens for non-trivial input
                mask_tokens_l = self.language_mask_projector(
                    self.language_mask_token.unsqueeze(0).expand(B, -1)
                )
                language_emb = torch.where(
                    language_mask.unsqueeze(1),
                    mask_tokens_l,
                    language_emb
                )
        else:
            vision_mask = torch.zeros(B, dtype=torch.bool, device=device)
            language_mask = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Encode with skip connections
        vision_latent, vision_skips = self.vision_encoder(vision_emb)  # (B, universal_dim)
        language_latent, language_skips = self.language_encoder(language_emb)  # (B, universal_dim)
        
        # CRITICAL: Normalize at universal_dim BEFORE reduction
        vision_latent = self.post_enc_norm_v(vision_latent)
        language_latent = self.post_enc_norm_l(language_latent)
        
        # Store high-dim versions for contrastive loss (BEFORE reduction)
        vision_latent_high = vision_latent.clone()  # Clone to preserve original
        language_latent_high = language_latent.clone()
        
        # Apply dimension reduction if enabled (ONLY ONCE)
        distillation_loss = 0
        if self.use_dimension_reduction and self.vision_distiller is not None:
            if return_distillation_loss:
                vision_latent, v_distill_loss = self.vision_distiller(vision_latent, return_loss=True)
                language_latent, l_distill_loss = self.language_distiller(language_latent, return_loss=True)
                distillation_loss = v_distill_loss + l_distill_loss
            else:
                vision_latent = self.vision_distiller(vision_latent)
                language_latent = self.language_distiller(language_latent)
            
            # Normalize after reduction for stability
            vision_latent = self.post_reduction_norm_v(vision_latent)
            language_latent = self.post_reduction_norm_l(language_latent)
        
        # Store clean latents BEFORE adding type embeddings (for contrastive loss)
        vision_latent_clean = vision_latent.clone()
        language_latent_clean = language_latent.clone()
        
        # Add type embeddings AFTER dimension reduction (now both at same dimension)
        vision_latent = vision_latent + self.vision_type_embedding.unsqueeze(0)
        language_latent = language_latent + self.language_type_embedding.unsqueeze(0)
        
        # Cross-modal fusion (type embeddings already added above)
        vision_fused, language_fused = self.cross_modal_fusion(vision_latent, language_latent)
        
        # Decode with skip connections (decoder handles dimension mismatch)
        vision_recon = self.vision_decoder(vision_fused, vision_skips, mask=vision_mask)
        language_recon = self.language_decoder(language_fused, language_skips, mask=language_mask)
        
        # Compute reconstruction losses IN ORIGINAL EMBEDDING SPACE (as per boss's notes)
        vision_loss = F.mse_loss(vision_recon, vision_original, reduction='none').mean(dim=1)
        language_loss = F.mse_loss(language_recon, language_original, reduction='none').mean(dim=1)
        
        # Simple batch mean for loss
        vision_loss = vision_loss.mean()
        language_loss = language_loss.mean()
        
        # Apply loss weighting
        total_loss = vision_loss + language_loss_weight * language_loss
        
        outputs = {
            'loss': total_loss,
            'vision_loss': vision_loss,
            'language_loss': language_loss,
            'vision_recon': vision_recon,
            'language_recon': language_recon,
            'vision_latent': vision_latent,
            'language_latent': language_latent,
            'vision_latent_clean': vision_latent_clean,  # WITHOUT type embeddings
            'language_latent_clean': language_latent_clean,  # WITHOUT type embeddings
            'vision_latent_high': vision_latent_high,
            'language_latent_high': language_latent_high,
        }
        
        if return_distillation_loss and self.use_dimension_reduction:
            outputs['distillation_loss'] = distillation_loss
            
        return outputs


# Keep the rest of the code (Dataset, training functions, etc.) the same
class DeepEarthDataset(Dataset):
    """
    Efficient dataset class that loads all data at initialization.
    Similar to the approach used in train_classifier.py.
    """
    
    def __init__(self, observation_ids: List[str], cache, device: str = 'cpu', 
                 load_batch_size: int = 64):
        """
        Initialize dataset by loading all data into memory.
        
        Args:
            observation_ids: List of observation IDs
            cache: UnifiedDataCache instance
            device: PyTorch device
            load_batch_size: Batch size for loading data
        """
        self.observation_ids = observation_ids
        self.device = device
        
        # Load all data at initialization
        self._load_all_data(cache, load_batch_size)
        
    def _load_all_data(self, cache, batch_size: int = 64):
        """Load all data into memory at once."""
        logger.info(f"Loading {len(self.observation_ids)} observations into memory...")
        
        # Check if we should use cached tensors
        cache_dir = Path("cached_embeddings")
        
        # Include more information in cache filename to prevent poisoning
        import hashlib
        # Hash the observation IDs to detect different splits
        obs_hash = hashlib.md5(''.join(sorted(self.observation_ids)).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"embeddings_{len(self.observation_ids)}_{batch_size}_{obs_hash}.pt"
        
        # TODO: For full-scale (4M observations), consider using np.memmap:
        # vision_mmap = np.memmap('vision.dat', dtype='float32', mode='w+', shape=(n_obs, 1408))
        # self.vision_embeddings = torch.from_numpy(vision_mmap)
        
        if cache_file.exists():
            logger.info(f"Loading pre-cached embeddings from {cache_file}")
            try:
                cached_data = torch.load(cache_file)
                self.vision_embeddings = cached_data['vision_embeddings']
                self.language_embeddings = cached_data['language_embeddings']
                self.species = cached_data['species']
                self.species_to_idx = cached_data['species_to_idx']
                self.idx_to_species = cached_data['idx_to_species']
                self.num_species = cached_data['num_species']
                logger.info(f"Successfully loaded cached data: {len(self.observation_ids)} observations, "
                           f"{self.num_species} unique species")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Loading from scratch...")
        
        all_vision_embs = []
        all_language_embs = []
        all_species = []
        
        # Load data in batches to manage memory
        for i in tqdm(range(0, len(self.observation_ids), batch_size), 
                      desc="Loading batches"):
            batch_ids = self.observation_ids[i:i + batch_size]
            
            try:
                batch_data = get_training_batch(
                    cache,
                    batch_ids,
                    include_vision=True,
                    include_language=True,
                    device='cpu'  # Load to CPU first
                )
                
                # Mean pool vision embeddings immediately to save memory
                # From (B, 8, 24, 24, 1408) to (B, 1408)
                vision_batch = batch_data['vision_embeddings']
                if vision_batch.dim() == 5:
                    vision_batch = vision_batch.mean(dim=(1, 2, 3))
                
                all_vision_embs.append(vision_batch)
                all_language_embs.append(batch_data['language_embeddings'])
                all_species.extend(batch_data['species'])
                
            except Exception as e:
                logger.warning(f"Error loading batch starting at {batch_ids[0]}: {e}")
                # Create dummy data for failed batch to maintain alignment
                dummy_vision = torch.zeros(len(batch_ids), 1408)  # Already mean-pooled shape
                dummy_language = torch.zeros(len(batch_ids), 7168)
                all_vision_embs.append(dummy_vision)
                all_language_embs.append(dummy_language)
                all_species.extend(['unknown'] * len(batch_ids))
        
        # Log before concatenation
        logger.info("Finished loading batches. Starting concatenation...")
        logger.info(f"Number of vision batches: {len(all_vision_embs)}")
        logger.info(f"Number of language batches: {len(all_language_embs)}")
        
        # Log memory usage estimate
        vision_memory_mb = sum(t.numel() * 4 / 1024 / 1024 for t in all_vision_embs)
        language_memory_mb = sum(t.numel() * 4 / 1024 / 1024 for t in all_language_embs)
        logger.info(f"Estimated memory usage - Vision: {vision_memory_mb:.1f} MB, Language: {language_memory_mb:.1f} MB")
        
        # Concatenate all embeddings
        logger.info("Concatenating vision embeddings...")
        self.vision_embeddings = torch.cat(all_vision_embs, dim=0)
        logger.info(f"Vision embeddings concatenated: {self.vision_embeddings.shape}")
        
        logger.info("Concatenating language embeddings...")
        self.language_embeddings = torch.cat(all_language_embs, dim=0)
        logger.info(f"Language embeddings concatenated: {self.language_embeddings.shape}")
        
        self.species = all_species
        
        # Create species mapping
        logger.info("Creating species mapping...")
        unique_species = sorted(list(set(all_species)))
        self.species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        self.num_species = len(unique_species)
        
        logger.info(f"Dataset loaded: {len(self.observation_ids)} observations, "
                   f"{self.num_species} unique species")
        logger.info(f"Vision shape: {self.vision_embeddings.shape}")
        logger.info(f"Language shape: {self.language_embeddings.shape}")
        
        # Save to cache for next time
        logger.info(f"Saving embeddings to cache for faster future loading...")
        cache_dir.mkdir(exist_ok=True)
        torch.save({
            'vision_embeddings': self.vision_embeddings,
            'language_embeddings': self.language_embeddings,
            'species': self.species,
            'species_to_idx': self.species_to_idx,
            'idx_to_species': self.idx_to_species,
            'num_species': self.num_species
        }, cache_file)
        logger.info(f"Cached embeddings saved to {cache_file}")
        
        logger.info("Data loading complete!")
        
    def __len__(self):
        return len(self.observation_ids)
    
    def __getitem__(self, idx):
        """Get single sample from dataset."""
        return {
            'vision_embedding': self.vision_embeddings[idx],
            'language_embedding': self.language_embeddings[idx],
            'obs_id': self.observation_ids[idx],
            'species': self.species[idx]
        }


def visualize_embeddings(original, reconstructed, name, epoch, save_dir, mask=None):
    """
    Visualize embeddings as 2D images with Turbo colormap
    
    Vision (1408D): Reshape to 32×44 for visualization
    Language (7168D): Reshape to 64×112 for visualization
    Universal (2048D): Reshape to 32×64 for visualization
    
    Args:
        original: Original embedding tensor
        reconstructed: Reconstructed embedding tensor
        name: Name for the visualization (Vision/Language/Universal)
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        mask: Optional mask tensor to show masked regions
    """
    # Create figure with subplots
    n_subplots = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots, 5))
    if n_subplots == 2:
        axes = [axes[0], axes[1], None]
    
    # Determine shape based on embedding size
    embedding_size = original.shape[0]
    if embedding_size == 1408:
        # Vision: 32×44 = 1408
        h, w = 32, 44
    elif embedding_size == 7168:
        # Language: 64×112 = 7168  
        h, w = 64, 112
    elif embedding_size == 2048:
        # Universal: 32×64 = 2048
        h, w = 32, 64
    else:
        # Generic case - try to make roughly square
        h = int(np.sqrt(embedding_size))
        w = embedding_size // h
        if h * w < embedding_size:
            w += 1
        logger.warning(f"Non-standard embedding size {embedding_size}, using {h}×{w} visualization")
    
    # Reshape tensors
    orig_2d = original[:h*w].view(h, w).cpu().numpy()
    recon_2d = reconstructed[:h*w].view(h, w).cpu().numpy()
    
    # Normalize to [0, 1]
    orig_norm = (orig_2d - orig_2d.min()) / (orig_2d.max() - orig_2d.min() + 1e-8)
    recon_norm = (recon_2d - recon_2d.min()) / (recon_2d.max() - recon_2d.min() + 1e-8)
    
    # Plot original
    im1 = axes[0].imshow(orig_norm, cmap='turbo', aspect='auto', interpolation='nearest')
    axes[0].set_title(f'{name} Original\nShape: {h}×{w}')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Normalized Value', fontsize=8)
    
    # Plot reconstructed
    im2 = axes[1].imshow(recon_norm, cmap='turbo', aspect='auto', interpolation='nearest')
    axes[1].set_title(f'{name} Reconstructed\nMSE: {F.mse_loss(original, reconstructed).item():.6f}')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Normalized Value', fontsize=8)
    
    # Plot mask if provided
    if mask is not None and axes[2] is not None:
        # Expand mask to match embedding dimensions if needed
        if mask.dim() == 0:  # Single boolean
            mask_2d = torch.full((h, w), mask.item(), dtype=torch.float32)
        else:
            mask_expanded = mask.float().unsqueeze(-1).expand(-1, embedding_size)
            mask_2d = mask_expanded[0, :h*w].view(h, w)
        
        im3 = axes[2].imshow(mask_2d.cpu().numpy(), cmap='gray', aspect='auto', 
                            interpolation='nearest', vmin=0, vmax=1)
        axes[2].set_title(f'Mask\n{mask.float().mean().item():.1%} masked')
        axes[2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        cbar3.set_label('Masked', fontsize=8)
    
    plt.suptitle(f'{name} Embeddings - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save with descriptive filename
    save_path = save_dir / f'{name.lower()}_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also save just the reconstructed image for animation
    fig_single = plt.figure(figsize=(6, 5))
    plt.imshow(recon_norm, cmap='turbo', aspect='auto', interpolation='nearest')
    plt.title(f'{name} Reconstruction - Epoch {epoch}')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    anim_path = save_dir / 'animations' / name.lower()
    anim_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(anim_path / f'frame_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_universal_embeddings(model, loader, device, epoch, save_dir, max_samples=5):
    """
    Visualize universal 2048D embeddings and their cross-modal alignment
    
    Shows:
    1. Vision → Universal (2048D) embedding
    2. Language → Universal (2048D) embedding  
    3. Cross-modal similarity in universal space
    """
    model.eval()
    save_dir = Path(save_dir)
    universal_dir = save_dir / 'universal_embeddings'
    universal_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        # Get first batch
        batch = next(iter(loader))
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Get universal embeddings (no masking for visualization)
        outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
        
        # Extract universal embeddings (2048D)
        vision_universal = outputs['vision_latent_high']  # Before dimension reduction
        language_universal = outputs['language_latent_high']
        
        # Visualize first few samples
        for i in range(min(max_samples, vision.shape[0])):
            # Create figure with universal space visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Vision universal embedding (32×64)
            v_universal = vision_universal[i].cpu()
            v_universal_2d = v_universal.view(32, 64).numpy()
            v_universal_norm = (v_universal_2d - v_universal_2d.min()) / (v_universal_2d.max() - v_universal_2d.min() + 1e-8)
            
            im1 = axes[0, 0].imshow(v_universal_norm, cmap='turbo', aspect='auto')
            axes[0, 0].set_title(f'Vision → Universal (2048D)\nSample {i+1}')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            
            # Language universal embedding (32×64)
            l_universal = language_universal[i].cpu()
            l_universal_2d = l_universal.view(32, 64).numpy()
            l_universal_norm = (l_universal_2d - l_universal_2d.min()) / (l_universal_2d.max() - l_universal_2d.min() + 1e-8)
            
            im2 = axes[0, 1].imshow(l_universal_norm, cmap='turbo', aspect='auto')
            axes[0, 1].set_title(f'Language → Universal (2048D)\nSample {i+1}')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
            
            # Difference map
            diff = np.abs(v_universal_norm - l_universal_norm)
            im3 = axes[1, 0].imshow(diff, cmap='hot', aspect='auto')
            axes[1, 0].set_title('Absolute Difference\n(Vision - Language)')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
            
            # Cosine similarity
            cosine_sim = F.cosine_similarity(v_universal, l_universal, dim=0).item()
            axes[1, 1].text(0.5, 0.5, f'Cosine Similarity\n{cosine_sim:.4f}', 
                           ha='center', va='center', fontsize=24, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Universal Space Visualization - Epoch {epoch}', fontsize=16)
            plt.tight_layout()
            
            save_path = universal_dir / f'universal_sample_{i+1}_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()


def create_embedding_animation(save_dir, name, fps=10):
    """
    Create animation from saved frames using matplotlib
    
    Args:
        save_dir: Base directory containing animations folder
        name: Name of the embedding type (vision/language)
        fps: Frames per second for animation
    """
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    
    anim_dir = Path(save_dir) / 'animations' / name.lower()
    if not anim_dir.exists():
        logger.warning(f"Animation directory {anim_dir} not found")
        return
    
    # Get all frames
    frames = sorted(list(anim_dir.glob('frame_*.png')))
    if not frames:
        logger.warning(f"No frames found in {anim_dir}")
        return
    
    # Load frames
    images = []
    for frame_path in frames:
        img = plt.imread(str(frame_path))
        images.append(img)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    im = ax.imshow(images[0])
    
    def animate(i):
        im.set_array(images[i])
        epoch = int(frames[i].stem.split('_')[1])
        ax.set_title(f'{name} Reconstruction - Epoch {epoch}', fontsize=14)
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(images), 
                                  interval=1000/fps, blit=True)
    
    # Save as GIF
    output_path = save_dir / f'{name.lower()}_reconstruction_animation.gif'
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer)
    plt.close()
    
    logger.info(f"Animation saved to {output_path}")


def visualize_embedding_evolution(save_dir, max_epochs=None):
    """
    Create a grid showing embedding evolution over epochs
    """
    save_dir = Path(save_dir)
    
    for embedding_type in ['vision', 'language']:
        # Find all epoch files
        pattern = f'{embedding_type}_epoch_*.png'
        epoch_files = sorted(list(save_dir.glob(pattern)))
        
        if not epoch_files:
            continue
            
        # Sample epochs to show (max 10 for readability)
        if max_epochs and len(epoch_files) > max_epochs:
            indices = np.linspace(0, len(epoch_files)-1, max_epochs, dtype=int)
            epoch_files = [epoch_files[i] for i in indices]
        
        n_epochs = len(epoch_files)
        if n_epochs == 0:
            continue
            
        # Create grid
        fig, axes = plt.subplots(2, (n_epochs+1)//2, figsize=(4*((n_epochs+1)//2), 8))
        axes = axes.flatten() if n_epochs > 1 else [axes]
        
        for i, epoch_file in enumerate(epoch_files):
            if i >= len(axes):
                break
                
            img = plt.imread(str(epoch_file))
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Extract epoch number from filename
            epoch_num = int(epoch_file.stem.split('_')[-1])
            axes[i].set_title(f'Epoch {epoch_num}', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(epoch_files), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{embedding_type.capitalize()} Embedding Evolution', fontsize=16)
        plt.tight_layout()
        
        evolution_path = save_dir / f'{embedding_type}_evolution_grid.png'
        plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evolution grid saved to {evolution_path}")


def visualize_latent_space(model, loader, device, epoch, save_dir, max_batches=10):
    """Visualize latent space representations"""
    model.eval()
    
    vision_latents = []
    language_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:  # Only use first N batches for visualization
                break
                
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            
            # Get latent representations
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            
            vision_latents.append(outputs['vision_latent'].cpu())
            language_latents.append(outputs['language_latent'].cpu())
    
    # Early return if no data
    if not vision_latents:
        return
    
    # Concatenate all latents
    vision_latents = torch.cat(vision_latents, dim=0).numpy()
    language_latents = torch.cat(language_latents, dim=0).numpy()
    
    # Plot latent distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vision latent
    axes[0].hist(vision_latents.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title('Vision Universal Embedding Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    # Language latent
    axes[1].hist(language_latents.flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_title('Language Universal Embedding Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    
    plt.suptitle(f'Universal Embedding Space Analysis - Epoch {epoch}')
    plt.tight_layout()
    
    save_path = save_dir / f'universal_space_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, loader, optimizer, device, scaler=None, scheduler=None,
                vision_mask_ratio=0.0, language_mask_ratio_train=1.00, language_loss_weight=1.0,
                contrastive_loss_fn=None, regularizer=None, 
                contrastive_weight=0.1, regularization_weight=0.01,
                use_distillation=False, distillation_weight=0.0):
    """Train for one epoch with MSE + optional contrastive loss and regularization"""
    model.train()
    
    # Sanity check masking ratio
    assert 0.0 <= language_mask_ratio_train <= 1.0, f"Invalid language mask ratio: {language_mask_ratio_train}"
    assert 0.0 <= vision_mask_ratio <= 1.0, f"Invalid vision mask ratio: {vision_mask_ratio}"
    
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    total_contrastive_loss = 0
    total_regularization_loss = 0
    total_contrastive_acc_v2l = 0
    total_contrastive_acc_l2v = 0
    
    use_amp = scaler is not None
    skipped_batches = 0  # Track skipped batches
    
    for i, batch in enumerate(loader):
        try:
            # Move batch to device with pin_memory benefits
            vision = batch['vision_embedding'].to(device, non_blocking=True)
            language = batch['language_embedding'].to(device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if use_amp:
                from torch.cuda.amp import autocast
                # Use bfloat16 when supported (faster than FP8 on H100 with Flash Attention)
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    # Fallback to float16
                    dtype = torch.float16
                    
                with autocast(dtype=dtype):
                    outputs = model(vision, language, 
                                  vision_mask_ratio=vision_mask_ratio, 
                                  language_mask_ratio=language_mask_ratio_train, 
                                  language_loss_weight=language_loss_weight,
                                  return_distillation_loss=False)
                    
                    # Base MSE loss
                    loss = outputs['loss']
                    
                    # Add contrastive loss if available
                    if contrastive_loss_fn is not None:
                        contrastive_loss, contrastive_metrics = contrastive_loss_fn(
                            outputs['vision_latent_clean'], 
                            outputs['language_latent_clean']
                        )
                        loss = loss + contrastive_weight * contrastive_loss
                        outputs['contrastive_loss'] = contrastive_loss
                        outputs['contrastive_metrics'] = contrastive_metrics
                    else:
                        outputs['contrastive_loss'] = torch.tensor(0.0, device=device)
                        outputs['contrastive_metrics'] = {'v2l_acc': 0.0, 'l2v_acc': 0.0}
                    
                    # Add regularization if available
                    if regularizer is not None and IMPROVEMENTS_AVAILABLE:
                        reg_loss, reg_losses = regularizer(
                            outputs['vision_latent_clean'],
                            outputs['language_latent_clean']
                        )
                        loss = loss + regularization_weight * reg_loss
                        outputs['regularization_loss'] = reg_loss
                    else:
                        outputs['regularization_loss'] = torch.tensor(0.0, device=device)
                        
            else:
                outputs = model(vision, language, 
                              vision_mask_ratio=vision_mask_ratio, 
                              language_mask_ratio=language_mask_ratio_train,
                              language_loss_weight=language_loss_weight,
                              return_distillation_loss=False)
                
                # Base MSE loss
                loss = outputs['loss']
                
                # Add contrastive loss if available
                if contrastive_loss_fn is not None:
                    contrastive_loss, contrastive_metrics = contrastive_loss_fn(
                        outputs['vision_latent_clean'], 
                        outputs['language_latent_clean']
                    )
                    loss = loss + contrastive_weight * contrastive_loss
                    outputs['contrastive_loss'] = contrastive_loss
                    outputs['contrastive_metrics'] = contrastive_metrics
                else:
                    outputs['contrastive_loss'] = torch.tensor(0.0, device=device)
                    outputs['contrastive_metrics'] = {'v2l_acc': 0.0, 'l2v_acc': 0.0}
                
                # Add regularization if available
                if regularizer is not None and IMPROVEMENTS_AVAILABLE:
                    reg_loss, reg_losses = regularizer(
                        outputs['vision_latent_clean'],
                        outputs['language_latent_clean']
                    )
                    loss = loss + regularization_weight * reg_loss
                    outputs['regularization_loss'] = reg_loss
                else:
                    outputs['regularization_loss'] = torch.tensor(0.0, device=device)
            
            # Check for NaN or extremely high loss
            if torch.isnan(loss) or loss.item() > 1e6:
                logger.warning(f"Abnormal loss detected at batch {i}: {loss.item()}")
                skipped_batches += 1
                continue
            
            # Backward with optional mixed precision
            optimizer.zero_grad()
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update scheduler if provided (step-based scheduling)
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate
            total_loss += loss.item()
            total_vision_loss += outputs['vision_loss'].item()
            total_language_loss += outputs['language_loss'].item()
            total_contrastive_loss += outputs['contrastive_loss'].item()
            total_regularization_loss += outputs['regularization_loss'].item()
            total_contrastive_acc_v2l += outputs['contrastive_metrics']['v2l_acc']
            total_contrastive_acc_l2v += outputs['contrastive_metrics']['l2v_acc']
            
            if i % 10 == 0:
                print(f"\r  Batch {i}/{len(loader)}: "
                      f"**Total={loss.item():.4f}**, "
                      f"MSE={outputs['loss'].item():.4f}, "
                      f"V={outputs['vision_loss'].item():.4f}, "
                      f"L={outputs['language_loss'].item():.4f}", 
                      end='', flush=True)
                if contrastive_loss_fn is not None:
                    print(f", C={outputs['contrastive_loss'].item():.4f}", end='', flush=True)
                
        except RuntimeError as e:
            logger.error(f"Error at batch {i}: {e}")
            if "CUDA" in str(e):
                logger.error("CUDA error detected, trying to recover...")
                torch.cuda.empty_cache()
            raise
    
    print()
    
    # Count actual batches processed
    n_batches = i + 1 - skipped_batches
    
    # Warn if too many batches were skipped
    if skipped_batches > 0:
        skip_rate = skipped_batches / (i + 1)
        logger.warning(f"Skipped {skipped_batches}/{i+1} batches ({skip_rate:.1%})")
        if skip_rate > 0.005:  # More than 0.5%
            logger.warning("High batch skip rate detected! Check model stability.")
    
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches,
        'regularization_loss': total_regularization_loss / n_batches,
        'distillation_loss': 0.0,  # Not used
        'contrastive_acc_v2l': total_contrastive_acc_v2l / n_batches,
        'contrastive_acc_l2v': total_contrastive_acc_l2v / n_batches
    }


def evaluate_comprehensive(model, train_loader, test_loader, device, epoch, save_dir, config_path, train_dataset, test_dataset):
    """
    Comprehensive evaluation including:
    1. Vision→Language retrieval @1, @5, @10
    2. Language→Vision retrieval @1, @5, @10
    3. Nearest neighbor analysis
    4. Classification using universal embeddings with actual species labels
    
    FIXED: Properly handles many-to-one vision-to-language relationships
    """
    model.eval()
    
    # Extract all embeddings and track by obs_id AND species
    train_vision_embeds = []
    train_language_embeds = []
    train_obs_ids = []
    train_species = []  # Track species for each embedding
    
    test_vision_embeds = []
    test_language_embeds = []
    test_obs_ids = []
    test_species = []  # Track species for each embedding
    
    with torch.no_grad():
        # Get train embeddings
        for batch in train_loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            train_vision_embeds.append(outputs['vision_latent'])
            train_language_embeds.append(outputs['language_latent'])
            train_obs_ids.extend(batch['obs_id'])
            train_species.extend(batch['species'])  # Track species
        
        # Get test embeddings
        for batch in test_loader:
            vision = batch['vision_embedding'].to(device)
            language = batch['language_embedding'].to(device)
            outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
            test_vision_embeds.append(outputs['vision_latent'])
            test_language_embeds.append(outputs['language_latent'])
            test_obs_ids.extend(batch['obs_id'])
            test_species.extend(batch['species'])  # Track species
    
    # Concatenate (without normalization to avoid double regularization)
    train_vision = torch.cat(train_vision_embeds, dim=0)
    train_language = torch.cat(train_language_embeds, dim=0)
    test_vision = torch.cat(test_vision_embeds, dim=0) if test_vision_embeds else torch.empty(0, train_vision.shape[1])
    test_language = torch.cat(test_language_embeds, dim=0) if test_language_embeds else torch.empty(0, train_language.shape[1])
    
    results = {}
    
    # Create species to indices mapping for efficient retrieval
    def create_species_to_indices(species_list):
        """Create mapping from species to list of indices"""
        species_to_indices = {}
        for idx, species in enumerate(species_list):
            if species not in species_to_indices:
                species_to_indices[species] = []
            species_to_indices[species].append(idx)
        return species_to_indices
    
    # FIXED: Vision→Language Retrieval (accounting for many-to-one)
    print("\n  📊 Vision→Language Retrieval:")
    for split_name, vision_emb, language_emb, species_list in [
        ("Train", train_vision, train_language, train_species), 
        ("Test", test_vision, test_language, test_species)
    ]:
        if len(vision_emb) == 0:
            continue
            
        # Compute similarities
        vision_norm = F.normalize(vision_emb, p=2, dim=1)
        language_norm = F.normalize(language_emb, p=2, dim=1)
        similarity = torch.mm(vision_norm, language_norm.t())
        
        # Get top-k accuracies
        for k in [1, 5, 10]:
            if k > len(vision_emb):
                continue
            _, indices = similarity.topk(k, dim=1)
            
            # Check if ANY of the retrieved items has the correct species
            correct = 0
            for i in range(len(vision_emb)):
                gt_species = species_list[i]
                retrieved_indices = indices[i].cpu().tolist()
                retrieved_species = [species_list[j] for j in retrieved_indices]
                
                # Success if ground truth species appears in retrieved species
                if gt_species in retrieved_species:
                    correct += 1
                    
            acc = 100.0 * correct / len(vision_emb)
            results[f'{split_name}_V2L_R@{k}'] = acc
            print(f"    {split_name} R@{k}: {acc:.1f}%")
    
    # FIXED: Language→Vision Retrieval (accounting for many-to-one)
    print("\n  📊 Language→Vision Retrieval:")
    for split_name, vision_emb, language_emb, species_list in [
        ("Train", train_vision, train_language, train_species), 
        ("Test", test_vision, test_language, test_species)
    ]:
        if len(language_emb) == 0:
            continue
            
        # For L→V: compute retrieval per unique species (not per embedding)
        unique_species = list(set(species_list))
        species_to_indices = create_species_to_indices(species_list)
        
        # Get representative language embedding for each species
        # Use average of all embeddings for the species to avoid sampling bias
        species_language_embeds = {}
        for species in unique_species:
            species_indices = species_to_indices[species]
            # Average all language embeddings for this species
            species_embeds = language_emb[species_indices]
            species_language_embeds[species] = species_embeds.mean(dim=0, keepdim=True)
        
        # Stack all species embeddings
        species_lang_tensor = torch.cat([species_language_embeds[s] for s in unique_species], dim=0)
        
        # Compute similarities
        vision_norm = F.normalize(vision_emb, p=2, dim=1)
        species_lang_norm = F.normalize(species_lang_tensor, p=2, dim=1)
        similarity = torch.mm(species_lang_norm, vision_norm.t())
        
        # Evaluate retrieval for each unique species
        total_correct = {k: 0 for k in [1, 5, 10]}
        
        for i, species in enumerate(unique_species):
            # Get top-k retrieved vision indices
            for k in [1, 5, 10]:
                if k > len(vision_emb):
                    continue
                    
                _, retrieved_indices = similarity[i].topk(k)
                retrieved_species = [species_list[j] for j in retrieved_indices.cpu().tolist()]
                
                # Success if the species appears in retrieved results
                if species in retrieved_species:
                    total_correct[k] += 1
        
        # Compute accuracies
        for k in [1, 5, 10]:
            if k > len(vision_emb):
                continue
            acc = 100.0 * total_correct[k] / len(unique_species)
            results[f'{split_name}_L2V_R@{k}'] = acc
            print(f"    {split_name} R@{k}: {acc:.1f}% (over {len(unique_species)} unique species)")
    
    # Additional statistics about many-to-one relationships
    print("\n  📊 Dataset Statistics:")
    for split_name, species_list in [("Train", train_species), ("Test", test_species)]:
        if not species_list:
            continue
        unique_species = list(set(species_list))
        avg_images_per_species = len(species_list) / len(unique_species) if unique_species else 0
        
        # Find species with most images
        species_counts = {}
        for s in species_list:
            species_counts[s] = species_counts.get(s, 0) + 1
        if species_counts:
            max_species = max(species_counts.items(), key=lambda x: x[1])
            print(f"    {split_name}: {len(species_list)} images, {len(unique_species)} species")
            print(f"    Average images per species: {avg_images_per_species:.1f}")
            print(f"    Most common species: {max_species[0]} ({max_species[1]} images)")
    
    # 3. Species Classification Probe
    print("\n  🎯 Species Classification Probe:")
    
    # Create species labels using actual species names
    train_species_labels = []
    test_species_labels = []
    
    # Use the species mapping from the datasets
    species_to_idx = train_dataset.species_to_idx
    
    # Get species labels for the actual samples that were processed
    for species in train_species:
        train_species_labels.append(species_to_idx[species])
    
    for species in test_species:
        if species in species_to_idx:
            test_species_labels.append(species_to_idx[species])
        else:
            test_species_labels.append(0)  # Unknown species
    
    train_species_labels = torch.tensor(train_species_labels).to(device)
    test_species_labels = torch.tensor(test_species_labels).to(device)
    
    # Verify dimensions match
    assert len(train_vision) == len(train_species_labels), f"Train dimension mismatch: {len(train_vision)} vs {len(train_species_labels)}"
    assert len(test_vision) == len(test_species_labels), f"Test dimension mismatch: {len(test_vision)} vs {len(test_species_labels)}"
    
    # Train a simple linear classifier on universal embeddings
    num_species = len(species_to_idx)
    classifier = nn.Linear(train_vision.shape[1], num_species).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    
    # Quick training with frozen backbone
    model.eval()
    for _ in range(100):
        logits = classifier(train_vision.detach())
        loss = F.cross_entropy(logits, train_species_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    with torch.no_grad():
        train_logits = classifier(train_vision)
        train_pred = train_logits.argmax(dim=1)
        train_acc = (train_pred == train_species_labels).float().mean() * 100
        
        if len(test_vision) > 0:
            test_logits = classifier(test_vision)
            test_pred = test_logits.argmax(dim=1)
            test_acc = (test_pred == test_species_labels).float().mean() * 100
            print(f"    Train accuracy: {train_acc:.1f}% ({num_species} species)")
            print(f"    Test accuracy: {test_acc:.1f}%")
            results['train_species_acc'] = train_acc.item()
            results['test_species_acc'] = test_acc.item()
    
    # 4. Nearest Neighbor Analysis
    print("\n  🔍 Nearest Neighbor Analysis (Test set):")
    if len(test_vision) > 5 and len(test_species) > 5:
        # Normalize for similarity computation
        test_vision_norm = F.normalize(test_vision[:5], p=2, dim=1)
        train_vision_norm = F.normalize(train_vision, p=2, dim=1)
        
        # For first 5 test samples, find nearest neighbors in train set
        test_to_train_sim = torch.mm(test_vision_norm, train_vision_norm.t())
        
        for i in range(min(5, len(test_vision))):
            _, nn_indices = test_to_train_sim[i].topk(3)
            test_species_name = test_species[i]
            nn_species = []
            for nn_idx in nn_indices.cpu().tolist():
                if nn_idx < len(train_species):
                    nn_species.append(train_species[nn_idx])
            print(f"    Test {test_species_name} → NN: {nn_species}")
    
    # 5. Embedding Statistics
    print("\n  📈 Embedding Statistics:")
    print(f"    Train vision embedding mean norm: {train_vision.norm(dim=1).mean():.3f}")
    print(f"    Train language embedding mean norm: {train_language.norm(dim=1).mean():.3f}")
    if len(test_vision) > 0:
        print(f"    Test vision embedding mean norm: {test_vision.norm(dim=1).mean():.3f}")
        print(f"    Test language embedding mean norm: {test_language.norm(dim=1).mean():.3f}")
    
    # Save detailed results
    results_path = save_dir / f'evaluation_epoch_{epoch}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


@torch.no_grad()
def evaluate(model, loader, device, epoch=0, visualize=False, save_dir=None,
             vision_mask_ratio=0.0, language_mask_ratio_eval=0.75, language_loss_weight=1.0,
             contrastive_loss_fn=None, regularizer=None,
             contrastive_weight=0.1, regularization_weight=0.01):
    """Evaluate model with MSE + optional contrastive loss and regularization"""
    model.eval()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    total_contrastive_loss = 0
    total_regularization_loss = 0
    total_contrastive_acc_v2l = 0
    total_contrastive_acc_l2v = 0
    n_batches = 0
    
    # Store first batch for visualization
    first_batch_outputs = None
    first_batch_masks = None
    
    for batch in loader:
        # Move batch to device
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Generate masks for visualization
        B = vision.shape[0]
        vision_mask = torch.rand(B, device=device) < vision_mask_ratio
        language_mask = torch.rand(B, device=device) < language_mask_ratio_eval
        
        # Forward pass (evaluate with same masking as training)
        outputs = model(vision, language, 
                       vision_mask_ratio=vision_mask_ratio, 
                       language_mask_ratio=language_mask_ratio_eval, 
                       language_loss_weight=language_loss_weight)
        
        # Base MSE loss
        loss = outputs['loss']
        
        # Add contrastive loss if available
        if contrastive_loss_fn is not None:
            contrastive_loss, contrastive_metrics = contrastive_loss_fn(
                outputs['vision_latent_clean'], 
                outputs['language_latent_clean']
            )
            loss = loss + contrastive_weight * contrastive_loss
            outputs['contrastive_loss'] = contrastive_loss
            outputs['contrastive_metrics'] = contrastive_metrics
        else:
            outputs['contrastive_loss'] = torch.tensor(0.0, device=device)
            outputs['contrastive_metrics'] = {'v2l_acc': 0.0, 'l2v_acc': 0.0}
        
        # Add regularization if available
        if regularizer is not None and IMPROVEMENTS_AVAILABLE:
            reg_loss, reg_losses = regularizer(
                outputs['vision_latent_clean'],
                outputs['language_latent_clean']
            )
            loss = loss + regularization_weight * reg_loss
            outputs['regularization_loss'] = reg_loss
        else:
            outputs['regularization_loss'] = torch.tensor(0.0, device=device)
        
        total_loss += loss.item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
        total_contrastive_loss += outputs['contrastive_loss'].item()
        total_regularization_loss += outputs['regularization_loss'].item()
        total_contrastive_acc_v2l += outputs['contrastive_metrics']['v2l_acc']
        total_contrastive_acc_l2v += outputs['contrastive_metrics']['l2v_acc']
        
        if first_batch_outputs is None:
            first_batch_outputs = {
                'vision_original': vision[0].mean(dim=(0, 1, 2)) if vision.dim() == 5 else vision[0],
                'language_original': language[0],
                'vision_recon': outputs['vision_recon'][0],
                'language_recon': outputs['language_recon'][0],
                'vision_latent': outputs['vision_latent'][0],
                'language_latent': outputs['language_latent'][0],
                'vision_latent_high': outputs['vision_latent_high'][0],
                'language_latent_high': outputs['language_latent_high'][0]
            }
            first_batch_masks = {
                'vision_mask': vision_mask[0] if vision_mask_ratio > 0 else None,
                'language_mask': language_mask[0] if language_mask_ratio_eval > 0 else None
            }
        
        n_batches += 1
    
    # Visualize if requested
    if visualize and first_batch_outputs is not None and save_dir is not None:
        # Visualize native dimensionality embeddings with masks
        visualize_embeddings(
            first_batch_outputs['vision_original'],
            first_batch_outputs['vision_recon'],
            'Vision',
            epoch,
            save_dir,
            mask=first_batch_masks['vision_mask']
        )
        visualize_embeddings(
            first_batch_outputs['language_original'],
            first_batch_outputs['language_recon'],
            'Language',
            epoch,
            save_dir,
            mask=first_batch_masks['language_mask']
        )
        
        # Visualize universal embeddings (2048D)
        if model.use_dimension_reduction:
            # Show high-dimensional universal embeddings before reduction
            visualize_embeddings(
                first_batch_outputs['vision_latent_high'],
                first_batch_outputs['vision_latent_high'],  # No reconstruction at this level
                'Vision_Universal',
                epoch,
                save_dir
            )
            visualize_embeddings(
                first_batch_outputs['language_latent_high'],
                first_batch_outputs['language_latent_high'],  # No reconstruction at this level
                'Language_Universal',
                epoch,
                save_dir
            )
        
        # Also visualize latent space statistics
        visualize_latent_space(model, loader, device, epoch, save_dir)
        
        # Visualize universal space alignment
        visualize_universal_embeddings(model, loader, device, epoch, save_dir, max_samples=3)
    
    # Avoid division by zero
    if n_batches == 0:
        return {
            'loss': 0.0,
            'vision_loss': 0.0,
            'language_loss': 0.0,
            'contrastive_loss': 0.0,
            'regularization_loss': 0.0,
            'contrastive_acc_v2l': 0.0,
            'contrastive_acc_l2v': 0.0
        }
    
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches,
        'contrastive_loss': total_contrastive_loss / n_batches,
        'regularization_loss': total_regularization_loss / n_batches,
        'contrastive_acc_v2l': total_contrastive_acc_v2l / n_batches,
        'contrastive_acc_l2v': total_contrastive_acc_l2v / n_batches
    }


def load_splits(config_path, min_observations_per_species=5):
    """Load train/test observation IDs from config file, filtering by species frequency"""
    logger.info(f"Loading splits from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    # First, count observations per species
    species_counts = {}
    for obs_id, metadata in observation_mappings.items():
        species = metadata['taxon_name']
        if species not in species_counts:
            species_counts[species] = []
        species_counts[species].append(obs_id)
    
    # Filter species with enough observations
    valid_species = {species for species, obs_ids in species_counts.items() 
                     if len(obs_ids) >= min_observations_per_species}
    
    logger.info(f"Species with >={min_observations_per_species} observations: {len(valid_species)}/{len(species_counts)}")
    
    # Extract train and test observation IDs only for valid species
    train_obs_ids = []
    test_obs_ids = []
    
    for obs_id, metadata in observation_mappings.items():
        if metadata['taxon_name'] in valid_species:
            if metadata['split'] == 'train':
                train_obs_ids.append(obs_id)
            elif metadata['split'] == 'test':
                test_obs_ids.append(obs_id)
    
    logger.info(f"After filtering: {len(train_obs_ids)} train and {len(test_obs_ids)} test observations")
    
    # Log some statistics
    test_species = set([metadata['taxon_name'] for metadata in observation_mappings.values() 
                       if metadata['split'] == 'test' and metadata['taxon_name'] in valid_species])
    train_species = set([metadata['taxon_name'] for metadata in observation_mappings.values() 
                        if metadata['split'] == 'train' and metadata['taxon_name'] in valid_species])
    
    logger.info(f"Species in filtered test set: {len(test_species)}")
    logger.info(f"Species in filtered train set: {len(train_species)}")
    logger.info(f"Common species: {len(train_species & test_species)}")
    
    # Show examples of filtered species
    filtered_species = sorted([(s, len(obs)) for s, obs in species_counts.items() 
                              if len(obs) < min_observations_per_species], 
                             key=lambda x: x[1])
    if filtered_species:
        logger.info(f"Examples of filtered species: {filtered_species[:5]}")
    
    return train_obs_ids, test_obs_ids


def get_language_mask_ratio(epoch, warmup_epochs=5, target_ratio=1.0):
    """
    Warmup language masking from 0% to target over warmup_epochs.
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs to warm up masking
        target_ratio: Target masking ratio after warmup
    
    Returns:
        Current masking ratio
    """
    if warmup_epochs <= 0:
        return target_ratio
    
    if epoch < warmup_epochs:
        # Linear ramp from 0 to target
        return (epoch / warmup_epochs) * target_ratio
    else:
        return target_ratio


def get_contrastive_mining_enabled(epoch, disable_mining_epochs=10):
    """
    Disable hard negative mining for first N epochs.
    
    Args:
        epoch: Current epoch (0-indexed)
        disable_mining_epochs: Number of epochs to disable mining
    
    Returns:
        Whether to enable mining
    """
    return epoch >= disable_mining_epochs


def get_contrastive_weight(epoch, warmup_epochs=20, initial_weight=0.2, final_weight=0.8):
    """
    Schedule contrastive loss weight to increase after warmup.
    
    Args:
        epoch: Current epoch (0-indexed)
        warmup_epochs: Number of epochs to warm up with low weight
        initial_weight: Starting weight
        final_weight: Target weight after warmup
    
    Returns:
        Current contrastive weight
    """
    if epoch < warmup_epochs:
        return initial_weight
    else:
        # Linear ramp up over 10 epochs after warmup
        ramp_epochs = 10
        progress = min((epoch - warmup_epochs) / ramp_epochs, 1.0)
        return initial_weight + (final_weight - initial_weight) * progress


def main(deterministic_mode=False):
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    epochs = 100
    lr = 3e-4  # Keep higher learning rate
    num_workers = 0  # Set to 0 for simpler data loading like train_classifier
    warmup_epochs = 5  # Simple warmup for stability
    
    # Model architecture configuration
    universal_dim = 2048  # High-dim for initial learning
    use_dimension_reduction = False  # Keep it simple for now
    target_dim = 512  # Not used when dimension reduction is False
    use_vision_patches = True  # Use patch encoder for vision
    fusion_type = 'mlp'  # Simple fusion
    
    # UPDATED MASKING: Reduce to 75% for language
    vision_mask_ratio = 0.0  # NEVER touch vision
    language_mask_ratio = 0.75  # 75% masking - leaves 25% signal!
    language_loss_weight = 1.0  # Equal weight
    
    # RE-ENABLE CONTRASTIVE LOSS (prevents collapse)
    use_contrastive = True  # RE-ENABLED to prevent collapse
    contrastive_weight = 0.1  # Start low
    contrastive_temperature = 0.1  # Slightly higher for stability
    
    # ADD REGULARIZATION
    use_regularization = True and IMPROVEMENTS_AVAILABLE
    regularization_weight = 0.01  # Light regularization
    regularization_weights = {
        'orthogonality': 0.001,
        'diversity': 0.001,
        'alignment': 0.001
    }
    
    # Keep distillation disabled
    use_distillation = False
    
    # EARLY STOPPING CONFIGURATION
    patience = 250  # Stop if no improvement for 10 epochs
    min_delta = 0.001  # Minimum change to qualify as improvement
    
    # SPECIES FILTERING
    min_observations_per_species = 5  # Only use species with >= 5 observations
    
    # H100 optimizations
    if device == 'cuda':
        # Enable Flash Attention if available (PyTorch 2.0+)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash Attention enabled for H100")
        
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        
        # Check for FP8 support (H100 feature) - logged but not used
        supports_fp8 = hasattr(torch, 'float8_e4m3fn') and torch.cuda.get_device_capability()[0] >= 9
        if supports_fp8:
            logger.info("FP8 support detected on H100")
        
        # Use benchmark mode for better performance (unless deterministic mode requested)
        if not deterministic_mode:
            torch.backends.cudnn.benchmark = True
            logger.info("CuDNN benchmark mode enabled for better performance")
    
    logger.info(f"Using device: {device}")
    
    # Set CUDA environment variables to help with initialization
    if device == 'cuda' and deterministic_mode:
        # Set deterministic behavior only if requested
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic mode enabled (slower but reproducible)")
    
    # Change to dashboard directory for cache
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Load splits from config file
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        train_obs_ids, test_obs_ids = load_splits(config_path, min_observations_per_species)
        logger.info(f"Loaded splits: {len(train_obs_ids)} train, {len(test_obs_ids)} test (filtered for species with >={min_observations_per_species} observations)")
        
        # Create a single cache instance
        cache = UnifiedDataCache("dataset_config.json")
        logger.info("Created cache for data loading")
        
        # Create datasets using the efficient loading approach
        logger.info("Creating datasets...")
        train_dataset = DeepEarthDataset(
            observation_ids=train_obs_ids,
            cache=cache,
            device='cpu',  # Keep data on CPU, move to GPU during training
            load_batch_size=64
        )
        test_dataset = DeepEarthDataset(
            observation_ids=test_obs_ids,
            cache=cache,
            device='cpu',
            load_batch_size=64
        )
        
        logger.info(f"Datasets created with all data loaded into memory")
        
        # Create loaders - simpler without multiprocessing
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Force no workers to avoid pickling issues
            pin_memory=(device == 'cuda'),
            drop_last=False  # Don't drop last batch to keep all samples
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(batch_size, len(test_obs_ids)),
            shuffle=False,
            num_workers=0,  # Force no workers to avoid pickling issues
            pin_memory=(device == 'cuda'),
            drop_last=False  # Don't drop last batch
        )
        
        # Create model with new architecture options
        model = MultimodalMLPUNet(
            vision_dim=1408,
            language_dim=7168,
            universal_dim=universal_dim,
            use_vision_patches=use_vision_patches,
            fusion_type=fusion_type,
            use_dimension_reduction=use_dimension_reduction,
            target_dim=target_dim
        ).to(device)
        
        # Optional: Compile model for better performance (PyTorch 2.0+)
        compile_model = False  # Set to True to enable torch.compile
        if compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="max-autotune")
            logger.info("Model compiled successfully")
        
        # Initialize contrastive loss and regularizer
        if use_contrastive:
            if IMPROVEMENTS_AVAILABLE:
                from multimodal_improvements import ContrastiveLoss
                contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature)
                logger.info(f"Using ContrastiveLoss with temperature={contrastive_temperature}")
            else:
                contrastive_loss_fn = SimpleContrastiveLoss(temperature=contrastive_temperature)
                logger.info(f"Using SimpleContrastiveLoss with temperature={contrastive_temperature}")
        else:
            contrastive_loss_fn = None
            
        if use_regularization and IMPROVEMENTS_AVAILABLE:
            from multimodal_improvements import UniversalSpaceRegularizer
            regularizer = UniversalSpaceRegularizer(
                orthogonality_weight=regularization_weights['orthogonality'],
                diversity_weight=regularization_weights['diversity'],
                alignment_weight=regularization_weights['alignment']
            )
            logger.info(f"Regularization enabled with weights: {regularization_weights}")
        else:
            regularizer = None
        
        # Print type embedding norms (scaled to ~0.7 for better gradient flow)
        with torch.no_grad():
            vision_type_norm = model.vision_type_embedding.norm().item()
            language_type_norm = model.language_type_embedding.norm().item()
            print(f"Type embedding norms - Vision: {vision_type_norm:.3f}, Language: {language_type_norm:.3f}")
        
        # Optimizer with proper learning rate
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)  # Reduced weight decay
        
        # Calculate total steps for proper warmup
        steps_per_epoch = len(train_loader)
        warmup_steps = warmup_epochs * steps_per_epoch
        
        # Add learning rate scheduler with cosine annealing
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Warmup steps: {warmup_steps} ({warmup_epochs} epochs × {steps_per_epoch} batches/epoch)")
        
        # Setup visualization directory
        viz_dir = Path('visualizations')
        viz_dir.mkdir(exist_ok=True)
        
        # Create timestamped checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f'multimodal_mlp_unet_best_{timestamp}.pth'
        
        print("\n" + "="*80)
        print("🚀 MULTIMODAL AUTOENCODER WITH IMPROVED TRAINING")
        print("="*80)
        print(f"Architecture: MLP U-Net with concatenative skip connections")
        print(f"Universal embedding dimension: {universal_dim}")
        print(f"Vision encoder: {'Patch-based with attention aggregation' if use_vision_patches else 'Mean-pooled'}")
        print(f"Cross-modal fusion: {fusion_type}")
        print(f"\n📌 MASKING CONFIGURATION:")
        print(f"  Vision masking: {int(vision_mask_ratio*100)}% (NEVER TOUCHED)")
        print(f"  Language masking: {int(language_mask_ratio*100)}% (Reduced from 100%!)")
        print(f"  ✓ Masking raw 7168-D language embeddings BEFORE encoder")
        print(f"\n🎯 LOSS CONFIGURATION:")
        print(f"  MSE Reconstruction: Vision MSE + Language MSE")
        if use_contrastive:
            print(f"  Contrastive Loss: ENABLED (weight={contrastive_weight}, temp={contrastive_temperature})")
            print(f"    - Prevents representation collapse")
        if use_regularization:
            print(f"  Regularization: ENABLED (weight={regularization_weight})")
            print(f"    - Prevents overfitting")
        print(f"\n⚡ TRAINING IMPROVEMENTS:")
        print(f"  Learning rate: {lr} with cosine annealing")
        print(f"  Early stopping: patience={patience} epochs")
        print(f"  Gradient clipping: 1.0")
        print(f"  Species filtering: >={min_observations_per_species} observations per species")
        print(f"\nBatch size: {batch_size}")
        print(f"Dataset: {len(train_obs_ids)} train, {len(test_obs_ids)} test samples (after filtering)")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Mixed Precision: Enabled ({'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'})")
        print("="*80 + "\n")
        
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Initialize mixed precision scaler if using CUDA
        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda')) if device == 'cuda' else None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Fixed language masking at 75%
            current_language_mask_ratio =1.00 #language_mask_ratio
            
            # Train with MSE + contrastive + regularization
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, scaler, scheduler,
                vision_mask_ratio=vision_mask_ratio,
                language_mask_ratio_train=current_language_mask_ratio,
                language_loss_weight=language_loss_weight,
                contrastive_loss_fn=contrastive_loss_fn,
                regularizer=regularizer,
                contrastive_weight=contrastive_weight,
                regularization_weight=regularization_weight,
                use_distillation=False,
                distillation_weight=0.0
            )
            
            # BOLD MSE DISPLAY
            print(f"\n🔥 TRAIN METRICS:")
            print(f"   **VISION MSE: {train_metrics['vision_loss']:.6f}**")
            print(f"   **LANGUAGE MSE: {train_metrics['language_loss']:.6f}**")
            print(f"   **TOTAL LOSS: {train_metrics['loss']:.6f}**")
            if use_contrastive:
                print(f"   Contrastive: {train_metrics['contrastive_loss']:.4f} (V2L: {train_metrics['contrastive_acc_v2l']:.1%}, L2V: {train_metrics['contrastive_acc_l2v']:.1%})")
            if use_regularization:
                print(f"   Regularization: {train_metrics['regularization_loss']:.4f}")
            
            # Evaluate with both masked and unmasked settings
            visualize = (epoch % 5 == 0)
            
            # Masked evaluation (matches training)
            test_metrics_masked = evaluate(
                model, test_loader, device, epoch, visualize, viz_dir,
                vision_mask_ratio=vision_mask_ratio,
                language_mask_ratio_eval=current_language_mask_ratio,
                language_loss_weight=language_loss_weight,
                contrastive_loss_fn=contrastive_loss_fn,
                regularizer=regularizer,
                contrastive_weight=contrastive_weight,
                regularization_weight=regularization_weight
            )
            
            # Unmasked evaluation (pure reconstruction capacity)
            with torch.no_grad():
                test_metrics_unmasked = evaluate(
                    model, test_loader, device, epoch, False, None,
                    vision_mask_ratio=0.0,
                    language_mask_ratio_eval=0.0,
                    language_loss_weight=language_loss_weight,
                    contrastive_loss_fn=contrastive_loss_fn,
                    regularizer=regularizer,
                    contrastive_weight=contrastive_weight,
                    regularization_weight=regularization_weight
                )
            
            # BOLD MSE DISPLAY FOR TEST
            print(f"\n🔥 TEST METRICS (Masked):")
            print(f"   **VISION MSE: {test_metrics_masked['vision_loss']:.6f}**")
            print(f"   **LANGUAGE MSE: {test_metrics_masked['language_loss']:.6f}**")
            print(f"   **TOTAL LOSS: {test_metrics_masked['loss']:.6f}**")
            if use_contrastive:
                print(f"   Contrastive: {test_metrics_masked['contrastive_loss']:.4f} (V2L: {test_metrics_masked['contrastive_acc_v2l']:.1%}, L2V: {test_metrics_masked['contrastive_acc_l2v']:.1%})")
            
            print(f"\n🔥 TEST METRICS (Unmasked):")
            print(f"   **VISION MSE: {test_metrics_unmasked['vision_loss']:.6f}**")
            print(f"   **LANGUAGE MSE: {test_metrics_unmasked['language_loss']:.6f}**")
            print(f"   **TOTAL MSE: {test_metrics_unmasked['vision_loss'] + test_metrics_unmasked['language_loss']:.6f}**")
            
            # EARLY STOPPING CHECK
            if test_metrics_masked['loss'] < best_loss - min_delta:
                best_loss = test_metrics_masked['loss']
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_loss_masked': test_metrics_masked['loss'],
                    'test_loss_unmasked': test_metrics_unmasked['loss'],
                    'config': {
                        'universal_dim': universal_dim,
                        'use_vision_patches': use_vision_patches,
                        'fusion_type': fusion_type,
                        'language_mask_ratio': language_mask_ratio,
                        'use_contrastive': use_contrastive,
                        'contrastive_weight': contrastive_weight,
                        'use_regularization': use_regularization,
                        'regularization_weight': regularization_weight
                    }
                }, checkpoint_name)
                print(f"  ✓ Saved best model to {checkpoint_name}!")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs (best: {best_loss:.6f} at epoch {best_epoch+1})")
                
                if patience_counter >= patience:
                    print(f"\n⚠️  EARLY STOPPING: No improvement for {patience} epochs!")
                    print(f"Best model was at epoch {best_epoch+1} with loss {best_loss:.6f}")
                    break
            
            # Comprehensive evaluation every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print("\n🔬 Running comprehensive evaluation...")
                eval_results = evaluate_comprehensive(
                    model, train_loader, test_loader, device, epoch, viz_dir, config_path, 
                    train_dataset, test_dataset
                )
                
                # Run universal space analysis if available
                if IMPROVEMENTS_AVAILABLE and analyze_universal_space is not None:
                    print("\n📊 Analyzing universal embedding space...")
                    space_analysis = analyze_universal_space(
                        model, test_loader, device, viz_dir, epoch, max_samples=1000
                    )
                print()
            
            # Print current learning rate (after all steps in epoch)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")
            

        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print(f"Best test loss: {best_loss:.4f}")
        print(f"Model saved as: {checkpoint_name}")
        print("="*80)
        
        # Create animations and evolution grids
        print("\n🎬 Creating visualizations...")
        try:
            # Create animation for each embedding type
            for embedding_type in ['vision', 'language']:
                create_embedding_animation(viz_dir, embedding_type, fps=5)
            
            # Create evolution grids
            visualize_embedding_evolution(viz_dir, max_epochs=10)
            
            print("✓ Visualizations created successfully!")
        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")
        
    finally:
        os.chdir(original_dir)


def test_model_forward():
    """Quick unit test to verify model forward pass works correctly"""
    print("Running model forward pass test...")
    
    # Create model with dimension reduction
    model = MultimodalMLPUNet(
        vision_dim=1408,
        language_dim=7168,
        universal_dim=2048,
        use_vision_patches=False,  # Simpler for testing
        fusion_type='mlp',
        use_dimension_reduction=True,
        target_dim=512
    )
    
    # Test inputs
    batch_size = 4
    x_v = torch.randn(batch_size, 1408)
    x_l = torch.randn(batch_size, 7168)
    
    # Forward pass
    try:
        outputs = model(x_v, x_l, vision_mask_ratio=0.0, language_mask_ratio=0.5)
        
        # Check output shapes
        assert outputs['vision_recon'].shape == x_v.shape, f"Vision recon shape mismatch: {outputs['vision_recon'].shape} vs {x_v.shape}"
        assert outputs['language_recon'].shape == x_l.shape, f"Language recon shape mismatch: {outputs['language_recon'].shape} vs {x_l.shape}"
        assert outputs['vision_latent'].shape[1] == 512, f"Vision latent should be 512D, got {outputs['vision_latent'].shape[1]}"
        assert outputs['language_latent'].shape[1] == 512, f"Language latent should be 512D, got {outputs['language_latent'].shape[1]}"
        
        print("✓ Model forward pass test passed!")
        return True
    except Exception as e:
        print(f"✗ Model forward pass test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pure Reconstruction Multimodal Autoencoder")
    parser.add_argument('--deterministic', action='store_true', 
                       help='Enable deterministic mode for reproducibility (slower)')
    parser.add_argument('--batch', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--test', action='store_true',
                       help='Run unit test only')
    args = parser.parse_args()
    
    if args.test:
        test_model_forward()
    else:
        main(deterministic_mode=args.deterministic)
