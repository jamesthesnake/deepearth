#!/usr/bin/env python3
"""Test that the model initializes correctly with projection heads"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from deepearth_mlp_improved import ImprovedMLPUNet
import torch

# Test model initialization
print("Testing model initialization...")
model = ImprovedMLPUNet(
    vision_dim=1408,
    language_dim=7168,
    universal_dim=2048,
    projection_dim=256,
    dropout=0.3,
    lightweight=True
)

print("\nModel components:")
for name, module in model.named_children():
    print(f"  {name}: {module.__class__.__name__}")
    
# Check if projection heads exist
assert hasattr(model, 'vision_projector'), "vision_projector missing!"
assert hasattr(model, 'language_projector'), "language_projector missing!"
print("\n✓ Projection heads found!")

# Test forward pass
print("\nTesting forward pass...")
batch_size = 4
vision_input = torch.randn(batch_size, 1408)
language_input = torch.randn(batch_size, 7168)

outputs = model(vision_input, language_input, use_projector=True)
print("\nOutput keys:", list(outputs.keys()))
print("Vision universal shape:", outputs['vision_universal'].shape)
print("Language universal shape:", outputs['language_universal'].shape)

print("\n✓ Model working correctly!")
os.chdir(dashboard_path)
