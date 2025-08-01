#!/usr/bin/env python3
"""
End-to-End Visualization Pipeline for Multimodal Model
Shows the journey from raw image → JEPA features → deep embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def load_raw_image(observation_id: str, image_path: Path) -> np.ndarray:
    """Load the raw observation image"""
    # This is a placeholder - adapt to your data loading pipeline
    # You'll need to implement based on your actual image storage
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualize_jepa_features(jepa_embedding: torch.Tensor, 
                           spatial_dims: Tuple[int, int] = (24, 24),
                           temporal_dim: int = 8) -> Dict[str, np.ndarray]:
    """
    Visualize JEPA features with multiple analysis methods
    
    Args:
        jepa_embedding: (8, 24, 24, 1408) tensor
        spatial_dims: Spatial dimensions (H, W)
        temporal_dim: Temporal dimension
    
    Returns:
        Dictionary of visualizations
    """
    # Take middle temporal frame
    middle_frame = jepa_embedding[temporal_dim // 2]  # (24, 24, 1408)
    
    # 1. PCA visualization of spatial features
    H, W, C = middle_frame.shape
    features_flat = middle_frame.reshape(-1, C).cpu().numpy()  # (576, 1408)
    
    # Apply PCA to reduce to 3 components for RGB visualization
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_flat)  # (576, 3)
    
    # Normalize to [0, 1] for visualization
    features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    features_pca_img = features_pca.reshape(H, W, 3)
    
    # 2. Attention/activation heatmap (using norm of features)
    feature_norms = torch.norm(middle_frame, dim=-1).cpu().numpy()  # (24, 24)
    feature_norms = (feature_norms - feature_norms.min()) / (feature_norms.max() - feature_norms.min())
    
    # 3. Channel statistics visualization
    channel_means = middle_frame.mean(dim=(0, 1)).cpu().numpy()  # (1408,)
    channel_stds = middle_frame.std(dim=(0, 1)).cpu().numpy()  # (1408,)
    
    return {
        'pca_rgb': features_pca_img,
        'attention_map': feature_norms,
        'channel_means': channel_means,
        'channel_stds': channel_stds,
        'pca_explained_variance': pca.explained_variance_ratio_
    }

def visualize_deep_embeddings(embedding: torch.Tensor, 
                             dim: int = 2048,
                             reshape_dims: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    """
    Visualize deep embeddings with normalization and color mapping
    
    Args:
        embedding: (2048,) tensor
        dim: Embedding dimension
        reshape_dims: Optional dimensions to reshape to (H, W)
    
    Returns:
        Dictionary of visualizations
    """
    # Normalize between 0 and 1
    emb_normalized = embedding.cpu().numpy()
    emb_normalized = (emb_normalized - emb_normalized.min()) / (emb_normalized.max() - emb_normalized.min())
    
    # Apply sqrt for better visualization (compresses high values)
    emb_sqrt = np.sqrt(emb_normalized)
    
    # Auto-compute reshape dimensions if not provided
    if reshape_dims is None:
        # Try to make it as square as possible
        sqrt_dim = int(np.sqrt(dim))
        if sqrt_dim * sqrt_dim == dim:
            reshape_dims = (sqrt_dim, sqrt_dim)
        else:
            # Find factors close to square
            for h in range(int(np.sqrt(dim)), 0, -1):
                if dim % h == 0:
                    reshape_dims = (h, dim // h)
                    break
    
    # Reshape for 2D visualization
    if reshape_dims is not None and reshape_dims[0] * reshape_dims[1] == dim:
        emb_2d = emb_sqrt[:reshape_dims[0] * reshape_dims[1]].reshape(reshape_dims)
    else:
        # Fallback: pad to nearest square
        sqrt_dim = int(np.ceil(np.sqrt(dim)))
        padded = np.pad(emb_sqrt, (0, sqrt_dim**2 - dim), constant_values=0)
        emb_2d = padded.reshape(sqrt_dim, sqrt_dim)
    
    return {
        'normalized': emb_normalized,
        'sqrt_scaled': emb_sqrt,
        'reshaped_2d': emb_2d,
        'stats': {
            'mean': embedding.mean().item(),
            'std': embedding.std().item(),
            'min': embedding.min().item(),
            'max': embedding.max().item(),
            'norm': embedding.norm().item()
        }
    }

def create_visualization_pipeline(
    raw_image: np.ndarray,
    jepa_features: torch.Tensor,
    vision_embedding: torch.Tensor,
    language_embedding: torch.Tensor,
    vision_latent: torch.Tensor,
    language_latent: torch.Tensor,
    species_name: str,
    save_path: Optional[Path] = None
):
    """
    Create comprehensive visualization of the entire pipeline
    
    Args:
        raw_image: Original image (H, W, 3)
        jepa_features: JEPA features (8, 24, 24, 1408)
        vision_embedding: Vision embedding (1408,)
        language_embedding: Language embedding (7168,)
        vision_latent: Universal vision latent (2048,)
        language_latent: Universal language latent (2048,)
        species_name: Species name for title
        save_path: Optional path to save figure
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Multimodal Pipeline Visualization: {species_name}', fontsize=20, fontweight='bold')
    
    # 1. Raw Image
    ax_raw = fig.add_subplot(gs[0:2, 0])
    ax_raw.imshow(raw_image)
    ax_raw.set_title('Raw Image', fontsize=14, fontweight='bold')
    ax_raw.axis('off')
    
    # Arrow 1
    ax_arrow1 = fig.add_subplot(gs[0:2, 1])
    ax_arrow1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                      xycoords='axes fraction')
    ax_arrow1.text(0.5, 0.5, 'JEPA\nEncoder', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_arrow1.axis('off')
    
    # 2. JEPA Features Analysis
    jepa_viz = visualize_jepa_features(jepa_features)
    
    # PCA visualization
    ax_pca = fig.add_subplot(gs[0, 2])
    ax_pca.imshow(jepa_viz['pca_rgb'])
    ax_pca.set_title('JEPA PCA (RGB)', fontsize=12)
    ax_pca.axis('off')
    
    # Attention map
    ax_att = fig.add_subplot(gs[1, 2])
    im_att = ax_att.imshow(jepa_viz['attention_map'], cmap='hot')
    ax_att.set_title('Feature Activation Map', fontsize=12)
    ax_att.axis('off')
    plt.colorbar(im_att, ax=ax_att, fraction=0.046)
    
    # Channel statistics
    ax_channels = fig.add_subplot(gs[0:2, 3])
    x = np.arange(len(jepa_viz['channel_means']))
    ax_channels.plot(x[::10], jepa_viz['channel_means'][::10], 'b-', alpha=0.7, label='Mean')
    ax_channels.fill_between(x[::10], 
                            jepa_viz['channel_means'][::10] - jepa_viz['channel_stds'][::10],
                            jepa_viz['channel_means'][::10] + jepa_viz['channel_stds'][::10],
                            alpha=0.3, color='blue')
    ax_channels.set_title('JEPA Channel Statistics', fontsize=12)
    ax_channels.set_xlabel('Channel')
    ax_channels.set_ylabel('Activation')
    ax_channels.grid(True, alpha=0.3)
    
    # 3. Vision Embedding (1408D)
    vision_emb_viz = visualize_deep_embeddings(vision_embedding, dim=1408, reshape_dims=(44, 32))
    ax_vision_emb = fig.add_subplot(gs[2, 0:2])
    im_v = ax_vision_emb.imshow(vision_emb_viz['reshaped_2d'], cmap='turbo', aspect='auto')
    ax_vision_emb.set_title(f'Vision Embedding (1408D)\nMean: {vision_emb_viz["stats"]["mean"]:.3f}, Norm: {vision_emb_viz["stats"]["norm"]:.1f}', fontsize=12)
    plt.colorbar(im_v, ax=ax_vision_emb)
    
    # 4. Language Embedding (7168D)
    lang_emb_viz = visualize_deep_embeddings(language_embedding, dim=7168, reshape_dims=(64, 112))
    ax_lang_emb = fig.add_subplot(gs[3, 0:2])
    im_l = ax_lang_emb.imshow(lang_emb_viz['reshaped_2d'], cmap='turbo', aspect='auto')
    ax_lang_emb.set_title(f'Language Embedding (7168D)\nMean: {lang_emb_viz["stats"]["mean"]:.3f}, Norm: {lang_emb_viz["stats"]["norm"]:.1f}', fontsize=12)
    plt.colorbar(im_l, ax=ax_lang_emb)
    
    # Arrow 2
    ax_arrow2 = fig.add_subplot(gs[2:4, 2])
    ax_arrow2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                      xycoords='axes fraction')
    ax_arrow2.text(0.5, 0.5, 'MLP\nU-Net\nEncoder', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_arrow2.axis('off')
    
    # 5. Universal Latent Space (2048D)
    vision_latent_viz = visualize_deep_embeddings(vision_latent, dim=2048, reshape_dims=(64, 32))
    ax_vision_latent = fig.add_subplot(gs[2, 3:5])
    im_vl = ax_vision_latent.imshow(vision_latent_viz['reshaped_2d'], cmap='turbo', aspect='auto')
    ax_vision_latent.set_title(f'Vision Latent (2048D)\nMean: {vision_latent_viz["stats"]["mean"]:.3f}, Norm: {vision_latent_viz["stats"]["norm"]:.1f}', fontsize=12)
    plt.colorbar(im_vl, ax=ax_vision_latent)
    
    lang_latent_viz = visualize_deep_embeddings(language_latent, dim=2048, reshape_dims=(64, 32))
    ax_lang_latent = fig.add_subplot(gs[3, 3:5])
    im_ll = ax_lang_latent.imshow(lang_latent_viz['reshaped_2d'], cmap='turbo', aspect='auto')
    ax_lang_latent.set_title(f'Language Latent (2048D)\nMean: {lang_latent_viz["stats"]["mean"]:.3f}, Norm: {lang_latent_viz["stats"]["norm"]:.1f}', fontsize=12)
    plt.colorbar(im_ll, ax=ax_lang_latent)
    
    # 6. Similarity visualization
    ax_sim = fig.add_subplot(gs[2:4, 5])
    
    # Compute cosine similarity
    v_norm = F.normalize(vision_latent.unsqueeze(0), p=2, dim=1)
    l_norm = F.normalize(language_latent.unsqueeze(0), p=2, dim=1)
    similarity = torch.mm(v_norm, l_norm.t()).item()
    
    # Create alignment visualization
    ax_sim.bar(['Vision-Language\nSimilarity'], [similarity], color='green' if similarity > 0.5 else 'orange')
    ax_sim.set_ylim([0, 1])
    ax_sim.set_ylabel('Cosine Similarity')
    ax_sim.set_title(f'Cross-Modal Alignment\n{similarity:.3f}', fontsize=12, fontweight='bold')
    ax_sim.grid(True, alpha=0.3)
    
    # Add information box
    info_text = f"""Pipeline Statistics:
    • JEPA: 8×24×24×1408 → 1408D
    • Vision: 1408D → 2048D
    • Language: 7168D → 2048D
    • PCA Variance: {jepa_viz['pca_explained_variance'][:3].sum():.1%}
    • Alignment: {similarity:.3f}"""
    
    ax_info = fig.add_subplot(gs[0:2, 4:6])
    ax_info.text(0.1, 0.5, info_text, fontsize=11, va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    ax_info.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return fig


def create_batch_embedding_comparison(
    embeddings_dict: Dict[str, torch.Tensor],
    species_list: list,
    save_path: Optional[Path] = None
):
    """
    Create a comparison visualization of multiple embeddings
    
    Args:
        embeddings_dict: Dictionary of embedding tensors
        species_list: List of species names
        save_path: Optional save path
    """
    n_samples = min(8, len(species_list))  # Show up to 8 samples
    
    fig, axes = plt.subplots(n_samples, len(embeddings_dict), figsize=(20, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        for j, (name, embeddings) in enumerate(embeddings_dict.items()):
            emb = embeddings[i]
            
            # Determine dimensions for reshaping
            if emb.shape[0] == 1408:
                reshape_dims = (44, 32)
            elif emb.shape[0] == 7168:
                reshape_dims = (64, 112)
            elif emb.shape[0] == 2048:
                reshape_dims = (64, 32)
            else:
                reshape_dims = None
            
            viz = visualize_deep_embeddings(emb, dim=emb.shape[0], reshape_dims=reshape_dims)
            
            im = axes[i, j].imshow(viz['reshaped_2d'], cmap='turbo', aspect='auto')
            
            if i == 0:
                axes[i, j].set_title(f'{name}\n{emb.shape[0]}D', fontsize=12, fontweight='bold')
            
            if j == 0:
                axes[i, j].set_ylabel(f'{species_list[i][:20]}...', fontsize=10)
            
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)
    
    plt.suptitle('Embedding Comparison Across Pipeline Stages', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


# Example usage function
def visualize_model_sample(model, data_sample, save_dir: Path):
    """
    Create visualization for a single sample through the model
    
    Args:
        model: Trained multimodal model
        data_sample: Dictionary with vision/language embeddings
        save_dir: Directory to save visualizations
    """
    save_dir.mkdir(exist_ok=True)
    
    # Get embeddings
    vision_emb = data_sample['vision_embedding']
    language_emb = data_sample['language_embedding']
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model(
            vision_emb.unsqueeze(0), 
            language_emb.unsqueeze(0),
            vision_mask_ratio=0.0,
            language_mask_ratio=0.0
        )
    
    # For now, use placeholder for raw image and JEPA features
    # In practice, you'd load these from your data pipeline
    raw_image = np.random.rand(224, 224, 3)  # Placeholder
    jepa_features = torch.randn(8, 24, 24, 1408)  # Placeholder
    
    # Create visualization
    create_visualization_pipeline(
        raw_image=raw_image,
        jepa_features=jepa_features,
        vision_embedding=vision_emb,
        language_embedding=language_emb,
        vision_latent=outputs['vision_latent'][0],
        language_latent=outputs['language_latent'][0],
        species_name=data_sample.get('species', 'Unknown'),
        save_path=save_dir / f"pipeline_viz_{data_sample.get('obs_id', 'sample')}.png"
    )
    
    # Create batch comparison
    create_batch_embedding_comparison(
        embeddings_dict={
            'Vision Input': vision_emb.unsqueeze(0),
            'Language Input': language_emb.unsqueeze(0),
            'Vision Latent': outputs['vision_latent'],
            'Language Latent': outputs['language_latent']
        },
        species_list=[data_sample.get('species', 'Unknown')],
        save_path=save_dir / f"embedding_comparison_{data_sample.get('obs_id', 'sample')}.png"
    )


if __name__ == "__main__":
    print("Multimodal Visualization Pipeline ready!")
    print("Use create_visualization_pipeline() to visualize the full pipeline")
    print("Use create_batch_embedding_comparison() to compare embeddings")
