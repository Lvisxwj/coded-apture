# MST Model Integration for HSI Reconstruction

This directory contains the adapted MST (Multi-Scale Transformer) model for hyperspectral image reconstruction in your CASSI (Coded Aperture Snapshot Spectral Imaging) project.

## Files Created:

### Core Components:
1. **MST.py** - Original MST model architecture (already existed)
2. **mst_dataset_adapted.py** - Dataset loader adapted for your project structure
3. **mst_norm.py** - Normalization utilities for HSI data
4. **mst_loss.py** - Loss functions optimized for HSI reconstruction
5. **mst_train_adapted.py** - Training script following your project conventions
6. **mst_test_adapted.py** - Testing script with comprehensive metrics

## Key Adaptations Made:

### 1. Dataset Integration (`mst_dataset_adapted.py`)
- **Input**: Uses your existing HSI data structure (84-channel hyperspectral images)
- **CASSI Simulation**: Maintains your coded aperture and spectral shifting simulation
- **Initial Reconstruction**: Provides initial HSI estimate for MST input using shift-back operation
- **Output**: Returns (input_hsi, target_hsi, mask) tuples compatible with MST

### 2. Model Configuration
- **Input Channels**: 84 (matches your HSI data)
- **Output Channels**: 84 (full HSI reconstruction)
- **Architecture**: 3-stage encoder-decoder with Multi-Scale Attention Blocks
- **Mask Integration**: Uses your shifted mask data for guided reconstruction

### 3. Training Integration (`mst_train_adapted.py`)
- **Normalization Control**: HSI normalization applied in training script as requested
- **Loss Function**: Hybrid loss (Charbonnier + L1 + SSIM) matching your existing setup
- **Batch Size**: Reduced to 4 due to MST's memory requirements
- **GPU**: Configured for CUDA device 3 (matches your setup)
- **Logging**: Same format as your existing training logs

### 4. Testing & Metrics (`mst_test_adapted.py`)
- **Standard Metrics**: Loss functions, PSNR, SSIM
- **HSI-Specific Metrics**: Spectral Angle Mapper (SAM)
- **Visualization**: RGB composite generation for visual inspection
- **Results Saving**: .mat files and visualization images

## Usage Instructions:

### Training:
```bash
cd Model/MST
python mst_train_adapted.py
```

### Testing:
```bash
cd Model/MST
python mst_test_adapted.py
```

### Key Parameters to Adjust:

#### Data Paths (in both train and test scripts):
```python
hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
mask_path = "/data4/zhuo-file/mask_sim.mat"
```

#### Training Hyperparameters:
```python
learning_rate = 0.0004
batch_size = 4  # Adjust based on GPU memory
Max_epoch = 300
train_set = 2560
```

## Data Flow:

1. **Input Processing**:
   - Raw HSI: [256, 256, 84] → CASSI simulation → 2D measurement: [256, 422]
   - Initial reconstruction: [256, 422] → shift-back → Initial HSI: [256, 256, 84]

2. **MST Processing**:
   - Input: Initial HSI estimate [84, 256, 256] + Mask [84, 256, 310]
   - Output: Reconstructed HSI [84, 256, 256]

3. **Loss Computation**:
   - Normalization applied to both output and ground truth
   - Hybrid loss computation for gradient descent

## Model Architecture Summary:

```
Input: [bs, 84, 256, 256] + Mask [bs, 84, 256, 310]
├── Embedding: 84→84 channels
├── Encoder (3 stages):
│   ├── Stage 1: 84→168 channels, MSAB blocks
│   ├── Stage 2: 168→336 channels, MSAB blocks
│   └── Stage 3: 336→672 channels, MSAB blocks
├── Bottleneck: 672 channels, MSAB blocks
├── Decoder (3 stages):
│   ├── Stage 1: 672→336 channels, MSAB blocks
│   ├── Stage 2: 336→168 channels, MSAB blocks
│   └── Stage 3: 168→84 channels, MSAB blocks
└── Output: 84→84 channels + residual connection
Output: [bs, 84, 256, 256]
```

## Integration with Existing Project:

- **Maintains compatibility** with your existing data structure
- **Normalization control** in training/testing scripts (not in dataset)
- **Same loss functions** and training paradigm
- **Same file structure** and naming conventions
- **GPU configuration** matches your setup (cuda:3)

## Memory Requirements:

MST requires more GPU memory than U-Net variants due to attention mechanisms. If you encounter memory issues:
- Reduce batch_size (currently set to 4)
- Use gradient checkpointing
- Reduce MST dimensions (currently dim=84)

This integration allows you to directly compare MST performance with your existing U-Net models while maintaining the same experimental setup and evaluation metrics.