#!/bin/bash
# phased_training_pipeline.sh
# Complete phased training pipeline for multimodal autoencoder
# Usage: nohup bash phased_training_pipeline.sh > training_log.txt 2>&1 &

# Set up environment
set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
EXPERIMENT_NAME="phased_training_$(date +%Y%m%d_%H%M%S)"
BASE_DIR="experiments/${EXPERIMENT_NAME}"
LOG_DIR="${BASE_DIR}/logs"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/pipeline.log"
}

# Function to check if previous phase succeeded
check_phase_success() {
    local phase_name=$1
    local checkpoint_path=$2
    
    if [ ! -f "${checkpoint_path}" ]; then
        log "ERROR: Phase ${phase_name} failed - checkpoint not found at ${checkpoint_path}"
        exit 1
    fi
    
    # Check if training metrics look reasonable
    if [ -f "${CHECKPOINT_DIR}/${phase_name}/training_metrics_summary.json" ]; then
        python3 -c "
import json
with open('${CHECKPOINT_DIR}/${phase_name}/training_metrics_summary.json', 'r') as f:
    metrics = json.load(f)
    best = metrics.get('best_individual_metrics', {})
    
    # Phase-specific success criteria
    if '${phase_name}' == 'phase1':
        test_acc = best.get('test_acc', {}).get('value', 0)
        v2l_r1 = best.get('v2l_r1', {}).get('value', 0)
        if test_acc < 0.20 or v2l_r1 < 0.05:
            print(f'WARNING: Phase 1 metrics below threshold - Acc: {test_acc:.1%}, V2L: {v2l_r1:.1%}')
            exit(1)
    elif '${phase_name}' == 'phase2':
        v2l_r1 = best.get('v2l_r1', {}).get('value', 0)
        if v2l_r1 < 0.15:
            print(f'WARNING: Phase 2 V2L below 15%: {v2l_r1:.1%}')
            exit(1)
    
    print(f'Phase ${phase_name} metrics look good!')
"
        if [ $? -ne 0 ]; then
            log "WARNING: Phase ${phase_name} metrics below expected thresholds"
        fi
    fi
}

# Function to run a training phase
run_phase() {
    local phase_name=$1
    local phase_cmd=$2
    
    log "=========================================="
    log "Starting ${phase_name}"
    log "=========================================="
    
    # Save command for reproducibility
    echo "${phase_cmd}" > "${LOG_DIR}/${phase_name}_command.txt"
    
    # Run training
    eval "${phase_cmd}" 2>&1 | tee "${LOG_DIR}/${phase_name}.log"
    
    # Check if phase completed successfully
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "✅ ${phase_name} completed successfully"
    else
        log "❌ ${phase_name} failed with exit code ${PIPESTATUS[0]}"
        exit 1
    fi
}

# Main pipeline
log "Starting phased training pipeline: ${EXPERIMENT_NAME}"
log "Base directory: ${BASE_DIR}"

# Save git commit hash for reproducibility
git rev-parse HEAD > "${LOG_DIR}/git_commit.txt" 2>/dev/null || true

# ============================================================
# PHASE 1: Foundation (Simple model, no masking)
# ============================================================

# Create phase 1 directories
mkdir -p "${CHECKPOINT_DIR}/phase1"
mkdir -p "${LOG_DIR}"

PHASE1_CMD="python unet_mlp.py \
  --epochs 15 \
  --batch-size 256 \
  --mask-strategy none \
  --max-species 50 \
  --subset 3000 \
  --lambda-contrast 0.1 \
  --lambda-rec 0.0 \
  --lr 1e-4 \
  --gradient-accumulation 1 \
  --universal-dim 1024 \
  --hidden-dim 256 \
  --monitor-collapse \
  --save-dir ${CHECKPOINT_DIR}/phase1"

run_phase "Phase 1: Foundation" "${PHASE1_CMD}"
check_phase_success "phase1" "${CHECKPOINT_DIR}/phase1/autoencoder_best.pth"

# Quick health check
log "Running embedding health check after Phase 1..."
python3 -c "
import torch
import sys
sys.path.append('.')
from unet_mlp import *

# Load model and a few batches
checkpoint = torch.load('${CHECKPOINT_DIR}/phase1/autoencoder_best.pth')
print(f'Phase 1 Best Metrics:')
for metric, info in checkpoint.get('best_individual_metrics', {}).items():
    if isinstance(info['value'], float) and info['value'] <= 1:
        print(f'  {metric}: {info[\"value\"]:.2%} (epoch {info[\"epoch\"]})')
    else:
        print(f'  {metric}: {info[\"value\"]:.3f} (epoch {info[\"epoch\"]})')
"

# ============================================================
# PHASE 2: Alignment (Frozen classifier, focus on projections)
# ============================================================

# Create phase 2 directories
mkdir -p "${CHECKPOINT_DIR}/phase2"

PHASE2_CMD="python unet_mlp.py \
  --epochs 10 \
  --batch-size 256 \
  --mask-strategy language_only \
  --language-mask-ratio 0.2 \
  --max-species 50 \
  --subset 3000 \
  --lambda-contrast 0.1 \
  --lambda-rec 0.0 \
  --use-hard-negatives \
  --hard-neg-ratio 0.1 \
  --lr 5e-5 \
  --freeze-classifier \
  --train-only-projections \
  --contrast-ramp-epochs 10 \
  --contrast-ramp-start 0.1 \
  --contrast-ramp-end 0.7 \
  --monitor-collapse \
  --gradient-accumulation 1 \
  --universal-dim 1024 \
  --hidden-dim 256 \
  --resume-from ${CHECKPOINT_DIR}/phase1/autoencoder_best.pth \
  --save-dir ${CHECKPOINT_DIR}/phase2"

run_phase "Phase 2: Alignment" "${PHASE2_CMD}"
check_phase_success "phase2" "${CHECKPOINT_DIR}/phase2/autoencoder_best.pth"

# ============================================================
# PHASE 3: U-Net Architecture (Transfer weights, add capacity)
# ============================================================

# Create phase 3 directories
mkdir -p "${CHECKPOINT_DIR}/phase3"

log "Transferring weights from MLP to U-Net..."
python unet_mlp.py transfer \
  "${CHECKPOINT_DIR}/phase2/autoencoder_best.pth" \
  "${CHECKPOINT_DIR}/phase3/unet_init.pth" \
  2>&1 | tee "${LOG_DIR}/weight_transfer.log"

PHASE3_CMD="python unet_mlp.py \
  --epochs 10 \
  --batch-size 128 \
  --mask-strategy language_only \
  --language-mask-ratio 0.2 \
  --max-species 50 \
  --subset 3000 \
  --lambda-contrast 0.7 \
  --lambda-rec 0.05 \
  --use-hard-negatives \
  --hard-neg-ratio 0.2 \
  --use-unet \
  --unet-features 32 64 128 256 \
  --use-spatial-attention \
  --lr 3e-5 \
  --gradient-accumulation 2 \
  --universal-dim 1024 \
  --hidden-dim 256 \
  --monitor-collapse \
  --resume-from ${CHECKPOINT_DIR}/phase3/unet_init.pth \
  --save-dir ${CHECKPOINT_DIR}/phase3"

run_phase "Phase 3: U-Net" "${PHASE3_CMD}"
check_phase_success "phase3" "${CHECKPOINT_DIR}/phase3/autoencoder_best.pth"

# ============================================================
# PHASE 4: Full Dataset Fine-tuning (Optional - all species)
# ============================================================

if [ "${RUN_FULL_FINETUNE:-false}" = "true" ]; then
    log "=========================================="
    log "Phase 4: Full Dataset Fine-tuning"
    log "=========================================="
    
    # Create phase 4 directories
    mkdir -p "${CHECKPOINT_DIR}/phase4"
    
    PHASE4_CMD="python unet_mlp.py \
      --epochs 10 \
      --batch-size 64 \
      --mask-strategy language_only \
      --language-mask-ratio 0.3 \
      --max-species 224 \
      --lambda-contrast 0.7 \
      --lambda-rec 0.05 \
      --use-hard-negatives \
      --hard-neg-ratio 0.3 \
      --use-unet \
      --unet-features 32 64 128 256 \
      --use-spatial-attention \
      --use-arcface \
      --lr 1e-5 \
      --gradient-accumulation 4 \
      --universal-dim 1024 \
      --hidden-dim 256 \
      --monitor-collapse \
      --use-balanced-sampler \
      --resume-from ${CHECKPOINT_DIR}/phase3/autoencoder_best.pth \
      --save-dir ${CHECKPOINT_DIR}/phase4"
    
    run_phase "Phase 4: Full Dataset" "${PHASE4_CMD}"
fi

# ============================================================
# Final Summary and Visualization
# ============================================================

log "=========================================="
log "Training Pipeline Complete!"
log "=========================================="

# Generate final report
python3 -c "
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

base_dir = Path('${BASE_DIR}')
phases = ['phase1', 'phase2', 'phase3']
if (base_dir / 'checkpoints/phase4/training_metrics_summary.json').exists():
    phases.append('phase4')

# Collect metrics across phases
all_metrics = {}
for phase in phases:
    metrics_file = base_dir / f'checkpoints/{phase}/training_metrics_summary.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics[phase] = json.load(f)

# Create summary plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
metrics_to_plot = [
    ('test_acc', 'Test Accuracy', True),
    ('v2l_r1', 'V→L R@1', True),
    ('l2v_r1', 'L→V R@1', True),
    ('instance_alignment', 'Instance Alignment', False),
    ('avg_contrast_loss', 'Contrast Loss', False),
    ('train_knn', 'Train k-NN', True)
]

for idx, (metric, title, is_percentage) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    for phase in phases:
        if phase in all_metrics:
            best = all_metrics[phase].get('best_individual_metrics', {}).get(metric, {})
            value = best.get('value', 0)
            epoch = best.get('epoch', 0)
            
            if is_percentage:
                ax.bar(phase, value * 100, label=f'{value:.1%} (ep{epoch})')
                ax.set_ylabel('Percentage')
            else:
                ax.bar(phase, value, label=f'{value:.3f} (ep{epoch})')
            
            ax.text(phase, value * (100 if is_percentage else 1) * 0.5, 
                   f'{value:.1%}' if is_percentage else f'{value:.2f}',
                   ha='center', va='center', fontweight='bold')
    
    ax.set_title(title)
    ax.set_xlabel('Training Phase')

plt.suptitle('Phased Training Results', fontsize=16)
plt.tight_layout()
plt.savefig(base_dir / 'training_summary.png', dpi=150)

# Print final summary
print('\\n' + '='*60)
print('FINAL TRAINING SUMMARY')
print('='*60)

for phase in phases:
    if phase in all_metrics:
        print(f'\\n{phase.upper()}:')
        best = all_metrics[phase].get('best_individual_metrics', {})
        for metric in ['test_acc', 'v2l_r1', 'l2v_r1', 'instance_alignment']:
            if metric in best:
                value = best[metric]['value']
                if value <= 1:
                    print(f'  {metric}: {value:.2%}')
                else:
                    print(f'  {metric}: {value:.3f}')

print('\\nTraining artifacts saved to: ${BASE_DIR}')
print('  - Checkpoints: ${CHECKPOINT_DIR}/')
print('  - Logs: ${LOG_DIR}/')
print('  - Summary plot: ${BASE_DIR}/training_summary.png')
" 2>&1 | tee -a "${LOG_DIR}/final_summary.log"

# Send notification if configured (optional)
if command -v notify-send &> /dev/null; then
    notify-send "Training Complete" "Phased training pipeline finished: ${EXPERIMENT_NAME}"
fi

log "Pipeline completed successfully!"
log "Results saved to: ${BASE_DIR}"
