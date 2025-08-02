import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
from tqdm import tqdm
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class UNetEncoder(nn.Module):
    """U-Net style encoder with skip connections"""
    def __init__(self, input_dim, universal_dim=2048, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            # Auto-generate hidden dimensions
            hidden_dims = []
            current = input_dim
            while current > universal_dim:
                current = int(current * 0.75)  # Reduce by 25% each layer
                if current > universal_dim:
                    hidden_dims.append(current)
        
        self.blocks = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [universal_dim]
        
        for i in range(len(dims) - 1):
            self.blocks.append(MLPBlock(dims[i], dims[i+1]))
        
    def forward(self, x):
        skip_connections = []
        
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        
        return x, skip_connections


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections"""
    def __init__(self, universal_dim=2048, output_dim=1408, encoder_dims=None):
        super().__init__()
        
        if encoder_dims is None:
            # Mirror encoder architecture
            encoder_dims = []
            current = output_dim
            while current > universal_dim:
                current = int(current * 0.75)
                if current > universal_dim:
                    encoder_dims.append(current)
            encoder_dims = [output_dim] + encoder_dims + [universal_dim]
        
        # Decoder dimensions (reverse of encoder, excluding input dim)
        decoder_dims = encoder_dims[-2::-1]  # Reverse, skip the input dimension
        
        self.blocks = nn.ModuleList()
        self.skip_projectors = nn.ModuleList()
        
        # First block (no skip)
        self.blocks.append(MLPBlock(universal_dim, decoder_dims[0]))
        self.skip_projectors.append(None)
        
        # Blocks with skip connections
        for i in range(1, len(decoder_dims)):
            in_dim = decoder_dims[i-1]
            out_dim = decoder_dims[i]
            
            # Skip connection adds encoder dimension
            skip_dim = encoder_dims[-(i+1)]  # Get corresponding encoder dimension
            self.blocks.append(MLPBlock(in_dim + skip_dim, out_dim))
            
            # Projector in case skip dimension doesn't match
            if skip_dim != in_dim:
                self.skip_projectors.append(nn.Linear(skip_dim, in_dim))
            else:
                self.skip_projectors.append(None)
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(decoder_dims[-1], output_dim)
        
    def forward(self, x, skip_connections):
        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1]
        
        # First block (no skip)
        x = self.blocks[0](x)
        
        # Blocks with skip connections
        for i in range(1, len(self.blocks)):
            if i-1 < len(skip_connections):
                skip = skip_connections[i-1]
                
                # Project skip if needed
                if self.skip_projectors[i] is not None:
                    skip = self.skip_projectors[i](skip)
                
                # Concatenate skip connection
                x = torch.cat([x, skip], dim=-1)
            
            x = self.blocks[i](x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x


class TokenMLPUNetEncoder(nn.Module):
    """MLP U-Net encoder that processes token sequences"""
    def __init__(self, input_dim, universal_dim=2048, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [int(input_dim * 0.75), int(input_dim * 0.5)]
        
        # Ensure we end at universal_dim
        if not hidden_dims or hidden_dims[-1] != universal_dim:
            hidden_dims = hidden_dims + [universal_dim]
        
        self.blocks = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            # Token-wise MLP blocks
            self.blocks.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) token sequences
        Returns:
            encoded: (B, N, universal_dim)
            skips: List of intermediate activations
        """
        skips = []
        
        for block in self.blocks:
            # Apply block to each token
            x = block(x)
            skips.append(x)
        
        return x, skips


class TokenMLPUNetDecoder(nn.Module):
    """MLP U-Net decoder that reconstructs token sequences"""
    def __init__(self, universal_dim=2048, output_dim=1408, encoder_dims=None):
        super().__init__()
        
        if encoder_dims is None:
            encoder_dims = [output_dim, int(output_dim * 0.75), int(output_dim * 0.5), universal_dim]
        
        # Decoder dimensions (reverse of encoder)
        decoder_dims = encoder_dims[-2::-1]
        
        self.blocks = nn.ModuleList()
        
        # First block (no skip)
        self.blocks.append(nn.Sequential(
            nn.Linear(universal_dim, decoder_dims[0]),
            nn.LayerNorm(decoder_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        ))
        
        # Blocks with skip connections
        for i in range(1, len(decoder_dims)):
            in_dim = decoder_dims[i-1] + encoder_dims[-(i+1)]  # With skip
            out_dim = decoder_dims[i]
            
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU() if i < len(decoder_dims) - 1 else nn.Identity(),
                nn.Dropout(0.1) if i < len(decoder_dims) - 1 else nn.Identity()
            ))
        
        # Final projection
        self.output_proj = nn.Linear(decoder_dims[-1], output_dim)
        
    def forward(self, x, skips):
        """
        Args:
            x: (B, N, universal_dim)
            skips: List of skip connections from encoder
        Returns:
            reconstructed: (B, N, output_dim)
        """
        # Reverse skips but exclude the last one (which is the final encoder output)
        skips = skips[:-1][::-1]
        
        # First block (no skip)
        x = self.blocks[0](x)
        
        # Blocks with skip connections
        for i in range(1, len(self.blocks)):
            if i-1 < len(skips):
                # Concatenate skip connection
                x = torch.cat([x, skips[i-1]], dim=-1)
            x = self.blocks[i](x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x


class SharedTokenUNet(nn.Module):
    """Shared U-Net that processes token sequences from both modalities"""
    def __init__(self, universal_dim=2048, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1536, 1024, 512]
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        dims = [universal_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.encoder_blocks.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        # Bottleneck processing (mixes information between tokens)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        decoder_dims = hidden_dims[::-1] + [universal_dim]
        
        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i] if i == 0 else decoder_dims[i] + hidden_dims[-(i+1)]
            out_dim = decoder_dims[i+1]
            
            self.decoder_blocks.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU() if i < len(decoder_dims) - 2 else nn.Identity(),
                nn.Dropout(0.1) if i < len(decoder_dims) - 2 else nn.Identity()
            ))
    
    def forward(self, vision_tokens, language_tokens):
        """
        Process vision and language tokens separately through shared U-Net
        
        Args:
            vision_tokens: (B, N_v, 2048)
            language_tokens: (B, N_l, 2048)
        Returns:
            vision_out: (B, N_v, 2048)
            language_out: (B, N_l, 2048)
        """
        # Process each modality separately through encoder
        vision_skips = []
        language_skips = []
        
        v_x = vision_tokens
        l_x = language_tokens
        
        for block in self.encoder_blocks:
            v_x = block(v_x)
            l_x = block(l_x)
            vision_skips.append(v_x)
            language_skips.append(l_x)
        
        # Process through bottleneck
        v_x = self.bottleneck(v_x)
        l_x = self.bottleneck(l_x)
        
        # Decode with skips
        vision_skips = vision_skips[:-1][::-1]
        language_skips = language_skips[:-1][::-1]
        
        for i, block in enumerate(self.decoder_blocks):
            if i > 0 and i-1 < len(vision_skips):
                v_x = torch.cat([v_x, vision_skips[i-1]], dim=-1)
                l_x = torch.cat([l_x, language_skips[i-1]], dim=-1)
            
            v_x = block(v_x)
            l_x = block(l_x)
        
        return v_x, l_x


class SequenceMultimodalAutoencoder(nn.Module):
    """
    Multimodal autoencoder with proper masking at embedding level
    """
    def __init__(self, vision_dim=1408, language_dim=7168, universal_dim=2048,
                 vision_tokens=None, language_tokens=20):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.universal_dim = universal_dim
        self.language_tokens = language_tokens
        
        # For vision: when we reshape to tokens, each token will have this dimension
        self.vision_n_tokens = 24
        self.vision_token_dim = (vision_dim + self.vision_n_tokens - 1) // self.vision_n_tokens  # ~59
        
        # For language: when we reshape to tokens, each token will have this dimension
        self.language_token_dim = (language_dim + language_tokens - 1) // language_tokens  # ~359
        
        # Vision MLP U-Net encoder/decoder - operates on token dimension
        # FIXED: Don't include universal_dim in hidden_dims since TokenMLPUNetEncoder adds it
        vision_hidden_dims = [128, 256]  # Encoder will use [59, 128, 256, 2048]
        self.vision_encoder = TokenMLPUNetEncoder(self.vision_token_dim, universal_dim, vision_hidden_dims)
        # For decoder, we need to know the actual encoder dimensions including universal_dim
        vision_encoder_dims = [self.vision_token_dim, 128, 256, universal_dim]
        self.vision_decoder = TokenMLPUNetDecoder(universal_dim, self.vision_token_dim, vision_encoder_dims)
        
        # Language MLP U-Net encoder/decoder - operates on token dimension
        # FIXED: Don't include universal_dim in hidden_dims since TokenMLPUNetEncoder adds it
        language_hidden_dims = [512, 1024]  # Encoder will use [359, 512, 1024, 2048]
        self.language_encoder = TokenMLPUNetEncoder(self.language_token_dim, universal_dim, language_hidden_dims)
        # For decoder, we need to know the actual encoder dimensions including universal_dim
        language_encoder_dims = [self.language_token_dim, 512, 1024, universal_dim]
        self.language_decoder = TokenMLPUNetDecoder(universal_dim, self.language_token_dim, language_encoder_dims)
        
        # Shared U-Net for processing in universal space
        self.shared_unet = SharedTokenUNet(universal_dim, hidden_dims=[1536, 1024, 512])
        
        # Mask embeddings (full embedding size)
        # Vision mask embedding remains learnable
        self.vision_mask_embedding = nn.Parameter(torch.randn(1, vision_dim) * 0.02)
        
        # Language mask embedding is now FIXED random noise (not learnable)
        self.register_buffer('language_mask_embedding', torch.randn(1, language_dim) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def reshape_vision_to_tokens(self, vision_emb):
        """
        Reshape vision embedding to token sequences
        """
        if vision_emb.dim() == 5:
            # (B, 8, 24, 24, 1408) -> (B, 8*24*24, 1408)
            B, T, H, W, C = vision_emb.shape
            return vision_emb.view(B, T*H*W, C)
        elif vision_emb.dim() == 2:
            # (B, 1408) -> Create patches
            B, D = vision_emb.shape
            n_patches = self.vision_n_tokens
            patch_dim = self.vision_token_dim
            
            # Pad if necessary to match expected dimensions
            total_dim = n_patches * patch_dim
            if D < total_dim:
                padding = total_dim - D
                vision_emb = F.pad(vision_emb, (0, padding))
            
            return vision_emb[:, :total_dim].view(B, n_patches, patch_dim)
        else:
            raise ValueError(f"Unexpected vision shape: {vision_emb.shape}")
    
    def prepare_language_tokens(self, language_emb):
        """
        Prepare language tokens
        """
        if language_emb.dim() == 3:
            return language_emb
        elif language_emb.dim() == 2:
            # (B, 7168) -> (B, 20, 359)
            B, D = language_emb.shape
            n_tokens = self.language_tokens
            token_dim = self.language_token_dim
            
            # Pad if necessary to match expected dimensions
            total_dim = n_tokens * token_dim
            if D < total_dim:
                padding = total_dim - D
                language_emb = F.pad(language_emb, (0, padding))
            
            return language_emb[:, :total_dim].view(B, n_tokens, token_dim)
        else:
            raise ValueError(f"Unexpected language shape: {language_emb.shape}")
    
    def forward(self, vision_emb, language_emb, vision_mask_ratio=0.0, language_mask_ratio=1.0):
        B = vision_emb.shape[0]
        device = vision_emb.device
        
        # Store originals for loss computation
        vision_original = vision_emb.clone()
        language_original = language_emb.clone()
        
        # Apply masking at the embedding level (before reshaping to tokens)
        vision_mask = None
        language_mask = None
        
        if self.training and vision_mask_ratio > 0:
            # Mask entire vision embeddings
            vision_mask = torch.rand(B, device=device) < vision_mask_ratio
            vision_emb = torch.where(
                vision_mask.unsqueeze(-1),
                self.vision_mask_embedding.expand(B, -1),
                vision_emb
            )
        
        if self.training and language_mask_ratio > 0:
            # Mask entire language embeddings
            language_mask = torch.rand(B, device=device) < language_mask_ratio
            language_emb = torch.where(
                language_mask.unsqueeze(-1),
                self.language_mask_embedding.expand(B, -1),
                language_emb
            )
        
        # Convert to token sequences
        vision_tokens = self.reshape_vision_to_tokens(vision_emb)
        language_tokens = self.prepare_language_tokens(language_emb)
        
        # Encode to universal space with skips
        vision_universal, vision_skips = self.vision_encoder(vision_tokens)
        language_universal, language_skips = self.language_encoder(language_tokens)
        
        # Process through shared U-Net
        vision_processed, language_processed = self.shared_unet(vision_universal, language_universal)
        
        # Decode back to original dimensions
        vision_recon = self.vision_decoder(vision_processed, vision_skips)
        language_recon = self.language_decoder(language_processed, language_skips)
        
        # Reshape reconstructions back to original embedding shape
        if vision_original.dim() == 2:
            # Flatten token sequence back to original dimension
            # vision_recon is (B, n_tokens, token_dim)
            B = vision_recon.shape[0]
            vision_recon_flat = vision_recon.view(B, -1)[:, :self.vision_dim]
        else:
            # Handle 5D case if needed
            vision_recon_flat = vision_recon.view(B, -1)[:, :self.vision_dim]
        
        # Language reconstruction
        B = language_recon.shape[0]
        language_recon_flat = language_recon.view(B, -1)[:, :self.language_dim]
        
        # Compute losses in original embedding space
        vision_loss = F.mse_loss(vision_recon_flat, vision_original)
        language_loss = F.mse_loss(language_recon_flat, language_original)
        
        total_loss = vision_loss + language_loss
        
        return {
            'loss': total_loss,
            'vision_loss': vision_loss,
            'language_loss': language_loss,
            'vision_recon': vision_recon_flat,
            'language_recon': language_recon_flat,
            'vision_tokens': vision_tokens,
            'language_tokens': language_tokens,
            'vision_universal': vision_universal,
            'language_universal': language_universal,
            'vision_mask': vision_mask,
            'language_mask': language_mask
        }


class SimpleDataset(Dataset):
    """Simple dataset that returns pre-computed embeddings"""
    def __init__(self, vision_embeddings, language_embeddings, obs_ids):
        self.vision_embeddings = vision_embeddings
        self.language_embeddings = language_embeddings
        self.obs_ids = obs_ids
        
    def __len__(self):
        return len(self.obs_ids)
    
    def __getitem__(self, idx):
        return {
            'vision_embedding': self.vision_embeddings[idx],
            'language_embedding': self.language_embeddings[idx],
            'obs_id': self.obs_ids[idx]
        }


def visualize_embeddings(original, reconstructed, name, epoch, save_dir, mask=None):
    """Visualize embeddings as 2D images with fixed normalization scale"""
    # Determine shape based on embedding size
    embedding_size = original.shape[0]
    if embedding_size == 1408:
        h, w = 32, 44
    elif embedding_size == 7168:
        h, w = 64, 112
    elif embedding_size == 2048:
        h, w = 32, 64
    else:
        h = int(np.sqrt(embedding_size))
        w = embedding_size // h
        if h * w < embedding_size:
            w += 1
    
    # Calculate MSE
    mse = F.mse_loss(original, reconstructed).item()
    
    # Reshape tensors
    orig_2d = original[:h*w].view(h, w).cpu().numpy()
    recon_2d = reconstructed[:h*w].view(h, w).cpu().numpy()
    
    # Fixed normalization range for better visibility of small differences
    vmin, vmax = -1.0, 1.0
    
    # Create figure
    fig, axes = plt.subplots(1, 4 if mask is not None else 3, figsize=(24 if mask is not None else 18, 5))
    
    # Original
    im1 = axes[0].imshow(orig_2d, cmap='turbo', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{name} Original\nShape: {h}×{w}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Reconstructed
    im2 = axes[1].imshow(recon_2d, cmap='turbo', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{name} Reconstructed\nMSE: {mse:.6f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Difference map (original - reconstructed)
    diff = orig_2d - recon_2d
    # Use a diverging colormap centered at 0 for differences
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', interpolation='nearest', 
                         vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'Difference (Orig - Recon)\nMax abs diff: {max_diff:.4f}')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    # Mask if provided
    if mask is not None and len(axes) > 3:
        # For embedding-level mask, show as a bar
        mask_vis = np.ones((h, w)) * mask.float().cpu().numpy()
        im4 = axes[3].imshow(mask_vis, cmap='gray', aspect='auto', vmin=0, vmax=1)
        axes[3].set_title(f'Masked: {"Yes" if mask else "No"}')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3], fraction=0.046)
    
    plt.suptitle(f'{name} Embeddings - Epoch {epoch} (Color scale: ±1)', fontsize=14)
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'{name.lower()}_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save frame for animation
    anim_path = save_dir / 'animations' / name.lower()
    anim_path.mkdir(parents=True, exist_ok=True)
    
    # Save just reconstructed for animation with fixed scale
    fig_single = plt.figure(figsize=(6, 5))
    plt.imshow(recon_2d, cmap='turbo', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
    plt.title(f'{name} Reconstruction - Epoch {epoch}')
    plt.axis('off')
    plt.colorbar(fraction=0.046)
    plt.savefig(anim_path / f'frame_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save MSE value
    with open(anim_path / f'mse_epoch_{epoch:03d}.txt', 'w') as f:
        f.write(str(mse))


def visualize_universal_embeddings(model, loader, device, epoch, save_dir, max_samples=3):
    """Visualize universal 2048D embeddings and their cross-modal alignment"""
    model.eval()
    save_dir = Path(save_dir)
    universal_dir = save_dir / 'universal_embeddings'
    universal_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        # Get first batch
        batch = next(iter(loader))
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Get universal embeddings
        outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
        vision_universal = outputs['vision_universal']
        language_universal = outputs['language_universal']
        
        # Visualize first few samples
        for i in range(min(max_samples, vision.shape[0])):
            # Get mean universal embedding per sample
            v_universal_mean = vision_universal[i].mean(dim=0)  # Average over tokens
            l_universal_mean = language_universal[i].mean(dim=0)
            
            # Create figure with universal space visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Vision universal embedding (32×64)
            v_universal_2d = v_universal_mean.view(32, 64).cpu().numpy()
            v_universal_norm = (v_universal_2d - v_universal_2d.min()) / (v_universal_2d.max() - v_universal_2d.min() + 1e-8)
            
            im1 = axes[0, 0].imshow(v_universal_norm, cmap='turbo', aspect='auto')
            axes[0, 0].set_title(f'Vision → Universal (2048D)\nSample {i+1}')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
            
            # Language universal embedding (32×64)
            l_universal_2d = l_universal_mean.view(32, 64).cpu().numpy()
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
            cosine_sim = F.cosine_similarity(v_universal_mean, l_universal_mean, dim=0).item()
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
    """Create animation from saved frames"""
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
    
    # Load all images
    images = []
    for frame_path in frames:
        img = plt.imread(str(frame_path))
        images.append(img)
    
    if not images:
        return
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    im = ax.imshow(images[0])
    
    def animate(i):
        im.set_array(images[i])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(images), 
                                  interval=1000/fps, blit=True)
    
    # Save as GIF
    output_path = save_dir / f'{name.lower()}_reconstruction_animation.gif'
    writer = PillowWriter(fps=fps)
    
    try:
        anim.save(str(output_path), writer=writer)
        logger.info(f"Animation saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save animation: {e}")
    
    plt.close(fig)


def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(vision, language)
                loss = outputs['loss']
        else:
            outputs = model(vision, language)
            loss = outputs['loss']
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
    
    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches
    }


@torch.no_grad()
def evaluate(model, loader, device, epoch=0, visualize=False, save_dir=None):
    model.eval()
    total_loss = 0
    total_vision_loss = 0
    total_language_loss = 0
    
    first_batch = None
    
    for batch in loader:
        vision = batch['vision_embedding'].to(device)
        language = batch['language_embedding'].to(device)
        
        # Evaluate without masking to see reconstruction quality
        outputs = model(vision, language, vision_mask_ratio=0.0, language_mask_ratio=0.0)
        
        total_loss += outputs['loss'].item()
        total_vision_loss += outputs['vision_loss'].item()
        total_language_loss += outputs['language_loss'].item()
        
        if first_batch is None and visualize:
            first_batch = {
                'vision_original': vision[0],
                'language_original': language[0],
                'vision_recon': outputs['vision_recon'][0],
                'language_recon': outputs['language_recon'][0],
                'vision_mask': outputs['vision_mask'][0] if outputs['vision_mask'] is not None else None,
                'language_mask': outputs['language_mask'][0] if outputs['language_mask'] is not None else None
            }
    
    if visualize and first_batch is not None and save_dir is not None:
        visualize_embeddings(
            first_batch['vision_original'],
            first_batch['vision_recon'],
            'Vision',
            epoch,
            save_dir,
            first_batch['vision_mask']
        )
        visualize_embeddings(
            first_batch['language_original'],
            first_batch['language_recon'],
            'Language',
            epoch,
            save_dir,
            first_batch['language_mask']
        )
    
    n_batches = len(loader)
    metrics = {
        'loss': total_loss / n_batches,
        'vision_loss': total_vision_loss / n_batches,
        'language_loss': total_language_loss / n_batches
    }
    
    # Print MSE values clearly
    print("\n" + "="*50)
    print("EVALUATION MSE VALUES:")
    print("="*50)
    print(f"Vision MSE (1408D):    {metrics['vision_loss']:.6f}")
    print(f"Language MSE (7168D):  {metrics['language_loss']:.6f}")
    print(f"Total MSE:             {metrics['loss']:.6f}")
    print("="*50 + "\n")
    
    return metrics


def load_splits(config_path, min_observations_per_species=5, holdout_species_ratio=0.2):
    """Load train/test observation IDs from config file, filtering by species frequency
    
    Args:
        config_path: Path to config file
        min_observations_per_species: Minimum observations per species to include
        holdout_species_ratio: Fraction of species to hold out entirely for testing
    """
    logger.info(f"Loading splits from {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    observation_mappings = config['observation_mappings']
    
    # First, count observations per species
    species_counts = {}
    species_obs_mapping = {}
    for obs_id, metadata in observation_mappings.items():
        species = metadata['taxon_name']
        if species not in species_counts:
            species_counts[species] = []
            species_obs_mapping[species] = []
        species_counts[species].append(obs_id)
        species_obs_mapping[species].append((obs_id, metadata))
    
    # Filter species with enough observations
    valid_species_list = [species for species, obs_ids in species_counts.items() 
                          if len(obs_ids) >= min_observations_per_species]
    
    logger.info(f"Species with >={min_observations_per_species} observations: {len(valid_species_list)}/{len(species_counts)}")
    
    # Split species into train and holdout sets
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(valid_species_list)
    
    n_holdout = int(len(valid_species_list) * holdout_species_ratio)
    holdout_species = set(valid_species_list[:n_holdout])
    train_species = set(valid_species_list[n_holdout:])
    
    logger.info(f"Holding out {len(holdout_species)} species entirely for testing")
    logger.info(f"Training on {len(train_species)} species")
    
    # Extract train and test observation IDs
    train_obs_ids = []
    test_obs_ids = []
    holdout_obs_ids = []  # Completely unseen species
    
    for species, obs_list in species_obs_mapping.items():
        if species in holdout_species:
            # All observations from holdout species go to holdout test set
            for obs_id, metadata in obs_list:
                holdout_obs_ids.append(obs_id)
        elif species in train_species:
            # For training species, use original train/test split
            for obs_id, metadata in obs_list:
                if metadata['split'] == 'train':
                    train_obs_ids.append(obs_id)
                elif metadata['split'] == 'test':
                    test_obs_ids.append(obs_id)
    
    logger.info(f"Split summary:")
    logger.info(f"  - Train observations: {len(train_obs_ids)} (from {len(train_species)} species)")
    logger.info(f"  - Test observations (seen species): {len(test_obs_ids)}")
    logger.info(f"  - Test observations (unseen species): {len(holdout_obs_ids)} (from {len(holdout_species)} species)")
    
    # Return both test sets
    return train_obs_ids, test_obs_ids, holdout_obs_ids, train_species, holdout_species


class DeepEarthDataset(Dataset):
    """
    Efficient dataset class that loads all data at initialization.
    Handles vision mean-pooling and species mapping.
    """
    def __init__(self, observation_ids, cache, device='cpu', load_batch_size=64):
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
        
    def _load_all_data(self, cache, batch_size=64):
        """Load all data into memory at once."""
        logger.info(f"Loading {len(self.observation_ids)} observations into memory...")
        
        # Check if we should use cached tensors
        cache_dir = Path("cached_embeddings")
        cache_dir.mkdir(exist_ok=True)
        
        # Include hash of observation IDs to detect different splits
        import hashlib
        obs_hash = hashlib.md5(''.join(sorted(self.observation_ids)).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"embeddings_{len(self.observation_ids)}_{batch_size}_{obs_hash}.pt"
        
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
                from services.training_data import get_training_batch
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
                dummy_vision = torch.zeros(len(batch_ids), 1408)
                dummy_language = torch.zeros(len(batch_ids), 7168)
                all_vision_embs.append(dummy_vision)
                all_language_embs.append(dummy_language)
                all_species.extend(['unknown'] * len(batch_ids))
        
        # Concatenate all embeddings
        logger.info("Concatenating embeddings...")
        self.vision_embeddings = torch.cat(all_vision_embs, dim=0)
        self.language_embeddings = torch.cat(all_language_embs, dim=0)
        self.species = all_species
        
        # Create species mapping
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
        torch.save({
            'vision_embeddings': self.vision_embeddings,
            'language_embeddings': self.language_embeddings,
            'species': self.species,
            'species_to_idx': self.species_to_idx,
            'idx_to_species': self.idx_to_species,
            'num_species': self.num_species
        }, cache_file)
        logger.info(f"Cached embeddings saved to {cache_file}")
        
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


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    epochs = 50
    lr = 1e-3
    
    # Model configuration
    vision_dim = 1408
    language_dim = 7168
    universal_dim = 2048
    
    # Masking configuration
    vision_mask_ratio = 0.0  # No masking for vision
    language_mask_ratio = 1.0  # 100% masking for language
    
    # Species filtering
    min_observations_per_species = 5
    
    logger.info(f"Using device: {device}")
    
    # Add dashboard to path
    dashboard_path = Path(__file__).parent.parent / "dashboard"
    sys.path.insert(0, str(dashboard_path))
    
    from data_cache import UnifiedDataCache
    
    # Change to dashboard directory for cache
    original_dir = os.getcwd()
    os.chdir(dashboard_path)
    
    try:
        # Load splits from config file
        config_path = Path(__file__).parent / "config" / "central_florida_split.json"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return
            
        train_obs_ids, test_obs_ids, holdout_obs_ids, train_species, holdout_species = load_splits(
            config_path, min_observations_per_species, holdout_species_ratio=0.2
        )
        logger.info(f"Loaded splits: {len(train_obs_ids)} train, {len(test_obs_ids)} test (seen species), "
                   f"{len(holdout_obs_ids)} test (unseen species)")
        
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
        
        # Create holdout dataset for unseen species
        if len(holdout_obs_ids) > 0:
            holdout_dataset = DeepEarthDataset(
                observation_ids=holdout_obs_ids,
                cache=cache,
                device='cpu',
                load_batch_size=64
            )
            logger.info(f"Created holdout dataset with {len(holdout_obs_ids)} observations from unseen species")
        else:
            holdout_dataset = None
            logger.warning("No holdout observations available")
        
        logger.info(f"Datasets created with all data loaded into memory")
        
    finally:
        os.chdir(original_dir)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda'),
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(batch_size, len(test_obs_ids)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda'),
        drop_last=False
    )
    
    # Create holdout loader for unseen species
    holdout_loader = None
    if holdout_dataset is not None and len(holdout_obs_ids) > 0:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=min(batch_size, len(holdout_obs_ids)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda'),
            drop_last=False
        )
    
    # Create model
    model = SequenceMultimodalAutoencoder(
        vision_dim=vision_dim,
        language_dim=language_dim,
        universal_dim=universal_dim,
        language_tokens=20  # Approximate number of original language tokens
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Using sequence-preserving architecture with CORRECTED masking:")
    logger.info(f"- Vision masking: {vision_mask_ratio*100:.0f}% at embedding level")
    logger.info(f"- Language masking: {language_mask_ratio*100:.0f}% at embedding level")
    logger.info("- Shared U-Net for cross-modal interaction in universal space")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    # Visualization directory
    viz_dir = Path('visualizations_corrected_masking')
    viz_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("MULTIMODAL AUTOENCODER WITH CORRECTED MASKING")
    print("="*60)
    print(f"Vision: {vision_dim}D → {universal_dim}D → {vision_dim}D")
    print(f"Language: {language_dim}D → {universal_dim}D → {language_dim}D")
    print(f"Vision masking: {vision_mask_ratio*100:.0f}%")
    print(f"Language masking: {language_mask_ratio*100:.0f}% (FORCING CROSS-MODAL LEARNING)")
    print("="*60 + "\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scaler)
        print(f"\nTRAIN MSE:")
        print(f"  Vision MSE (1408D):   {train_metrics['vision_loss']:.6f}")
        print(f"  Language MSE (7168D): {train_metrics['language_loss']:.6f}")
        print(f"  Total MSE:            {train_metrics['loss']:.6f}")
        
        # Evaluate
        visualize = (epoch % 5 == 0) or (epoch == epochs - 1)
        test_metrics = evaluate(model, test_loader, device, epoch, visualize, viz_dir)
        
        # Evaluate on holdout species every 10 epochs
        if holdout_loader is not None and (epoch % 10 == 0 or epoch == epochs - 1):
            print("\n" + "="*50)
            print("EVALUATING ON UNSEEN SPECIES:")
            print("="*50)
            holdout_metrics = evaluate(model, holdout_loader, device, epoch, False, None)
            print(f"\nCOMPARISON:")
            print(f"Seen species - Vision MSE: {test_metrics['vision_loss']:.6f}, Language MSE: {test_metrics['language_loss']:.6f}")
            print(f"Unseen species - Vision MSE: {holdout_metrics['vision_loss']:.6f}, Language MSE: {holdout_metrics['language_loss']:.6f}")
            print(f"Language MSE ratio (unseen/seen): {holdout_metrics['language_loss']/test_metrics['language_loss']:.2f}x")
            print("="*50)
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_loss': test_metrics['loss'],
                'vision_mse': test_metrics['vision_loss'],
                'language_mse': test_metrics['language_loss']
            }, 'multimodal_autoencoder_best.pth')
            print(f"\n✓ Saved best model! (Total MSE: {best_loss:.6f})")
        
        print(f"\nLearning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best test loss: {best_loss:.6f}")
    print("="*60)
    
    # Create animations and visualizations
    print("\n🎬 Creating animations and visualizations...")
    try:
        # Create animations for each embedding type
        for embedding_type in ['vision', 'language']:
            create_embedding_animation(viz_dir, embedding_type, fps=5)
        
        # Visualize universal embeddings from last epoch
        visualize_universal_embeddings(model, test_loader, device, epochs-1, viz_dir, max_samples=3)
        
        print("✓ Visualizations created successfully!")
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
