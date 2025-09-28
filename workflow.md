# Project Workflow Documentation

This document describes the complete data flow and processing pipeline for the hyperspectral image reconstruction project.

## Overview Pipeline

```
Raw HSI Data [W,H,84] → CASSI Simulation → 2D Measurement [256,422] → Model → 32 Spectral Indices [32,256,256]
```

## Detailed Workflow

### 📥 **Step 1: Input Data Loading**

**Data Source:**
- **File Format**: `.mat` files containing hyperspectral images
- **File Location**: `/data4/zhuo-file/extracted_data/5nm_input/`
- **File Key**: `extracted_data` in .mat files
- **Total Files**: 210 files (skip corrupted: 15, 43, 109)

**Raw HSI Shape:**
```
Input HSI: [W, H, 84]
- W, H: Variable spatial dimensions (larger than 256×256)
- 84: Number of hyperspectral bands
- Value Range: ~455-13719 per band (see HSI_summary_results.txt)
```

**Ground Truth:**
- **File Location**: `/data4/zhuo-file/extracted_data/cor_r_label/`
- **File Format**: `{num}_indices.mat`
- **File Key**: `spectral_indices`
- **Shape**: `[W, H, 32]` - 32 precomputed spectral indices

### 🔄 **Step 2: Data Preprocessing**

**Random Cropping:**
```
HSI: [W, H, 84] → [256, 256, 84]
Target: [W, H, 32] → [256, 256, 32]
```

**Data Augmentation (Training Only):**
- Random rotation (0°, 90°, 180°, 270°)
- Random vertical flip
- Random horizontal flip

**Data Split:**
- **Training**: Files 0-200 (random selection)
- **Testing**: Files 200-250 (random selection)

### 🎭 **Step 3: CASSI Measurement Simulation**

**Coded Aperture Mask:**
```
Mask Shape: [256, 256] (binary mask)
Mask 3D: [256, 256, 84] (repeated for each spectral band)
Source: /data4/zhuo-file/mask_sim.mat
```

**Element-wise Masking:**
```
HSI [256,256,84] × Mask [256,256,84] → Masked HSI [256,256,84]
```

**Spectral Shifting (Dispersive Element Simulation):**
```
temp_shift: [256, 256+(84-1)×2, 84] = [256, 422, 84]
For each band t (0 to 83):
    temp_shift[:,:,t] = roll(temp_shift[:,:,t], shift=2×t, axis=1)
```

**Measurement Generation:**
```
2D Measurement = sum(temp_shift, axis=2) / 84 × 0.9
Final Shape: [256, 422]
```

### 🧠 **Step 4: Model Processing**

#### **U-Net Architecture:**

**Input Processing:**
```
Input: 2D Measurement [256, 422]
↓
initial_x() function: Shift-back operation
↓
Initial HSI Estimate [84, 256, 256]
```

**U-Net Encoder:**
```
Input: [84, 256, 256]
├── DoubleConv(84→64) → [64, 256, 256]
├── Down(64→128) → [128, 128, 128]
├── Down(128→256) → [256, 64, 64]
├── Down(256→512) → [512, 32, 32]
└── Down(512→1024) → [1024, 16, 16]
```

**U-Net Decoder:**
```
Bottleneck: [1024, 16, 16]
├── Up(1024→512) + Skip → [512, 32, 32]
├── Up(512→256) + Skip → [256, 64, 64]
├── Up(256→128) + Skip → [128, 128, 128]
├── Up(128→64) + Skip → [64, 256, 256]
└── OutConv(64→32) → [32, 256, 256]
```

**U-Net Parameters:**
- **Activation**: LeakyReLU (negative_slope=0.01)
- **Normalization**: BatchNorm2d
- **Skip Connections**: Concatenation + DoubleConv
- **Upsampling**: Bilinear interpolation (default)

#### **MST Architecture (Alternative):**

**Input Processing:**
```
Input: 2D Measurement [256, 422]
↓
initial_x() → Initial estimate → Feature extraction
↓
[dim, 256, 256] (dim=64 default)
```

**MST Encoder:**
```
Stage 1: [64, 256, 256] → MSAB → [128, 128, 128]
Stage 2: [128, 128, 128] → MSAB → [256, 64, 64]
Stage 3: [256, 64, 64] → MSAB → [512, 32, 32]
```

**MSAB (Multi-Scale Attention Block):**
- **MS_MSA**: Multi-Scale Multi-Head Self-Attention
- **FeedForward**: Conv2d layers with GELU activation
- **Num_blocks**: [2,2,2] attention blocks per stage

**MST Decoder:**
```
Bottleneck: [512, 32, 32] → MSAB
├── Up + Fusion → [256, 64, 64] → MSAB
├── Up + Fusion → [128, 128, 128] → MSAB
└── Up + Fusion → [64, 256, 256] → MSAB
```

**MST Output:**
```
[64, 256, 256] → Conv2d(64→32) → [32, 256, 256]
```

### 📊 **Step 5: Loss Computation & Normalization**

**Normalization (CRITICAL):**
```
Model Output: [32, 256, 256] (raw scale)
↓
max_min_norm(outputs) → [32, 256, 256] (0-1 scale)

Ground Truth: [32, 256, 256] (raw scale)
↓
max_min_norm(labels) → [32, 256, 256] (0-1 scale)
```

**Normalization Details:**
- **Function**: `max_min_norm()` from `Norm.py`
- **Method**: Per-channel min-max normalization
- **Formula**: `(data - min_values[i]) / (max_values[i] - min_values[i])`
- **Channels**: 32 precomputed min/max pairs for spectral indices

**Hybrid Loss Function:**
```
Total Loss = CharbonnierLoss + L1Loss + SSIMLoss

CharbonnierLoss: sqrt((pred-gt)² + eps²) - robust L1
L1Loss: |pred - gt| - mean absolute error
SSIMLoss: 1 - SSIM(pred, gt) - structural similarity
```

### 🎯 **Step 6: Output & Evaluation**

**Final Output Shape:**
```
32 Spectral Indices: [32, 256, 256]
- 32 channels: Different vegetation/material indices
- 256×256: Spatial resolution
- Value range: [0,1] after normalization
```

**Training Hyperparameters:**
```
Learning Rate: 0.0004
Batch Size: 8 (U-Net), 4 (MST)
Epochs: 500
Optimizer: AdamW(lr=0.0004, betas=(0.9,0.999))
Scheduler: MultiStepLR(milestones=[200,400], gamma=0.5)
GPU: CUDA device 3
```

**Model Saving:**
```
Path: /data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/parameter/ours/
Name: {Model}_lr{lr}_bs{bs}_Epoch{epoch}_{trainset}_hybridLoss.pth
```

### 📈 **Step 7: Testing & Metrics**

**Test Process:**
```
Test Split: Files 200-250
Batch Size: 1
Test Samples: 1200
Mode: model.eval() + torch.no_grad()
```

**Evaluation Metrics:**
- **CharbonnierLoss**: Robust reconstruction error
- **L1Loss**: Mean absolute error
- **SSIMLoss**: Structural similarity loss
- **Combined Loss**: Sum of all three losses

**Results Saving:**
```
Log File: {Model}_test_log.txt
Sample Results: .mat files with outputs and labels
Visualizations: RGB composite images (channels 0,1,2)
```

## Key Shape Transformations Summary

```
Raw HSI [W,H,84]
→ Crop [256,256,84]
→ CASSI [256,422]
→ Model [32,256,256]
→ Normalize [32,256,256]
→ Loss Computation
```

## Critical Pipeline Notes

1. **Normalization Bug Fixed**: Both model outputs and ground truth must be normalized before loss computation

2. **Dataset Returns**: `(input, target)` - 2 items only, not 3

3. **Data Split Control**: Hardcoded in `dataset.py:66-71` - modify here for different train/test splits

4. **Value Ranges**:
   - Raw HSI: ~455-13719 per band
   - 2D Measurement: ~0-15 range
   - Spectral Indices: Wide range (see Norm.py min/max values)
   - Normalized Output: [0,1] range

5. **Memory Requirements**: MST requires more GPU memory than U-Net due to attention mechanisms