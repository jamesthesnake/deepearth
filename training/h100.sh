#!/bin/bash
# Launch script for multimodal MLP U-Net training with H100 optimizations

# Disable NCCL P2P to avoid HBM copy penalty on H100
export NCCL_P2P_DISABLE=1

# Set CUDA device order by PCI bus ID for consistent GPU selection
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional: Set specific GPU
# export CUDA_VISIBLE_DEVICES=0

# Optional: Enable TF32 for better performance (H100 default)
export TORCH_ALLOW_TF32=1

# Optional: Set NCCL timeout for large models
export NCCL_TIMEOUT=1800

# Default batch size
BATCH_SIZE=${1:-128}

echo "Starting training with batch size: $BATCH_SIZE"
echo "Environment settings:"
echo "  NCCL_P2P_DISABLE=1 (avoiding HBM copy penalty)"
echo "  TORCH_ALLOW_TF32=1 (using TF32 for matmuls)"

# Run training
python complex_unet_animate.py --batch $BATCH_SIZE

# Alternative: Run with specific configurations
# python train_multimodal_mlp_unet.py --batch 256  # Even larger batch
# python train_multimodal_mlp_unet.py --batch 64 --deterministic  # Reproducible mode
