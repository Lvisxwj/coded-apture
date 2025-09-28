# Architecture Documentation

This document provides a detailed understanding of the codebase architecture, data flow, and component functionality for the hyperspectral image (HSI) processing project.

## Project Workflow Overview

```
Raw HSI Data (.mat files) → Dataset Loading → Measurement Simulation → Model Training/Testing → Reconstructed HSI Output
```

## Data Flow and Shape Transformations

### 1. Input Data Pipeline

#### Raw Data Format
- **Source**: .mat files containing hyperspectral images
- **Key**: `extracted_data` in .mat files
- **Original Shape**: `[W, H, 84]` where 84 is the number of spectral bands
- **Typical Size**: Variable W×H, with 84 spectral channels
- **Value Range**: Each layer has different min/max values (see `HSI_summary_results.txt`)

#### Data Loading Process (`train/dataset.py:64-135`)
1. **File Selection**: Loads 210 files, skips corrupted ones (15, 43, 109)
2. **Random Cropping**: Extracts 256×256 patches from larger images
3. **Data Augmentation**: Random rotation and flipping (training only)
4. **Mask Application**: Applies coded aperture mask simulation

#### Measurement Simulation (Critical Transformation)
```python
# Input: HSI [256, 256, 84]
# Output: Compressed measurement [256, 422]

# Key transformation in dataset.py:112-123
temp = mask_3d * hsi  # Apply coded aperture mask
temp_shift = np.zeros((256, 256 + (84-1)*2, 84))  # Create shifted array
# Apply spectral shifting (simulates dispersive element)
for t in range(84):
    temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2*t, axis=1)
meas = np.sum(temp_shift, axis=2)  # Sum across spectral dimension
input = meas / 84 * 0.9  # Normalize measurement
```

**Shape Evolution**:
- Original HSI: `[256, 256, 84]`
- Masked HSI: `[256, 256, 84]`
- Shifted array: `[256, 422, 84]` (422 = 256 + 83*2)
- Final measurement: `[256, 422]` (single 2D image)

### 2. Model Input/Output Shapes

#### U-Net Architecture (`Model/UNet.py`)

**Input Shape**: `[batch_size, 256, 422]` (compressed measurement)

**Internal Reconstruction Process**:
1. **Initial Reconstruction** (`initial_x`):
   - Input: `[bs, 256, 422]`
   - Output: `[bs, 84, 256, 256]` (reconstructed HSI cube)
   - Method: Reverse shift operation to extract spectral channels

2. **U-Net Processing**:
   - Encoder: 64 → 128 → 256 → 512 → 1024 channels
   - Decoder: 1024 → 512 → 256 → 128 → 64 → 32 channels
   - Skip connections preserve spatial information

**Final Output Shape**: `[batch_size, 32, 256, 256]` (32 spectral indices)

#### ATTU Architecture (`Model/ATTU.py`)
- Similar to U-Net but with attention mechanisms (MSABC blocks)
- Input/Output shapes identical to U-Net
- Additional transformer-based attention at each resolution level

### 3. Normalization Pipeline

#### Multiple Normalization Schemes (`train/Norm.py`)

**HSI Normalization** (for 84-channel data):
- **Function**: `HSI_max_min_norm()`
- **Input Shape**: `[bs, 84, W, H]`
- **Method**: Per-channel min-max normalization using precomputed statistics
- **Range**: Normalizes each spectral band to [0, 1] using band-specific min/max values

**Spectral Indices Normalization** (for 32-channel output):
- **Functions**: `max_min_norm()`, `XA_max_min_norm()`, `Chi_max_min_norm()`
- **Input Shape**: `[bs, 32, W, H]`
- **Purpose**: Different normalization schemes for different datasets/experiments
- **Method**: Channel-wise min-max normalization with dataset-specific statistics

#### Normalization Application Points:
1. **Training**: Only labels are normalized (`train.py:83`)
2. **Testing**: Only labels are normalized (`test.py:67`)
3. **Note**: Model outputs remain in original scale for easier interpretation

### 4. Loss Function Architecture (`train/loss.py`)

#### Hybrid Loss Combination:
```python
total_loss = CharbonnierLoss + L1Loss + SSIMLoss
```

**Individual Components**:
1. **Charbonnier Loss**: Robust L1-like loss with smooth gradients
2. **L1 Loss**: Mean absolute error
3. **SSIM Loss**: Structural similarity index (1 - SSIM score)

**Shape Requirements**: All losses expect `[bs, channels, H, W]` tensor pairs

## Key Functional Components

### 1. Dataset Class (`train/dataset.py`)

**Core Functionality**:
- **Initialization**: Loads ~207 HSI files and corresponding spectral indices
- **Data Splitting**: First 200 files for training, 200-250 for testing
- **Augmentation**: Random rotation, flipping, and cropping
- **Mask Simulation**: Applies coded aperture snapshot spectral imaging (CASSI) forward model

**Critical Parameters**:
- `train_set = 2560`: Number of training samples per epoch
- `size = 256`: Patch size for training
- `nums = 210`: Total number of data files to load

### 2. Model Architectures

#### U-Net Family
- **UNet**: Standard encoder-decoder with skip connections
- **ATTU**: U-Net with attention mechanisms and transformer blocks
- **UNet_KAN**: U-Net variant using Kolmogorov-Arnold Networks

#### Transformer Components
- **MS_MSA**: Multi-Scale Multi-Head Self-Attention
- **MSABC**: Multi-Scale Attention Block with CNN integration
- **TRM_Block**: Pure transformer blocks for feature processing

### 3. Training Pipeline (`train/train.py`)

**Training Loop**:
1. Load batch of compressed measurements and ground truth
2. Forward pass through model
3. Normalize ground truth labels
4. Compute hybrid loss
5. Backpropagate and update weights
6. Log metrics every iteration

**Key Training Parameters**:
- Batch size: 8
- Learning rate: 0.0004
- Epochs: 500
- GPU: CUDA device 3
- Optimizer: AdamW with MultiStepLR scheduler

### 4. Testing Pipeline (`test/test.py`)

**Evaluation Process**:
1. Load trained model weights
2. Process test dataset (batch size 1)
3. Generate reconstructions without gradients
4. Compute evaluation metrics
5. Save results for analysis

**Metrics Computed**:
- Charbonnier Loss
- L1 Loss
- SSIM Loss
- Combined loss

## Data Preprocessing Details

### Coded Aperture Mask
- **File**: `mask_sim.mat`
- **Shape**: `[256, 256]` binary mask
- **Purpose**: Simulates coded aperture in CASSI system
- **Application**: Element-wise multiplication with each spectral band

### Spectral Shifting Simulation
- **Purpose**: Simulates dispersive element in CASSI system
- **Method**: Each spectral band is shifted by `2*band_index` pixels horizontally
- **Result**: Creates overlapped spectral-spatial encoding

### Quality Control
- **Filter Rule**: Skip HSI files where any spectral band has max value > 16000
- **Corrupted Files**: Manually excluded files (15, 43, 109)
- **Range Analysis**: Precomputed via `Range_view.py` and stored in `HSI_summary_results.txt`

## Output Interpretation

### Reconstructed Spectral Indices
- **Channels**: 32 specific spectral indices (not full spectral reconstruction)
- **Physical Meaning**: Each channel represents a specific vegetation/material index
- **Applications**: Remote sensing, agriculture, environmental monitoring

### File Naming Convention
```
{Model_name}_lr{learning_rate}_bs{batch_size}_Epoch{epochs}_{train_set}_hybridLoss.pth
```

Example: `UNet_lr4-4_bs8_Epoch500_2560_hybridLoss.pth`

This architecture implements a complete CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction pipeline, transforming compressed 2D measurements back into spectral information using deep learning approaches.

## Critical Issues Discovered

### 1. **Dataset Return vs Training Loop Mismatch (CRITICAL BUG)**
- **Training Loop Expects**: `for i, (hsi, inputs, labels) in enumerate(train_loader)` (3 items)
- **Dataset Actually Returns**: `return input, target` (only 2 items)
- **Impact**: This would cause unpacking errors during training
- **Location**: `train/train.py:77` vs `train/dataset.py:135`

### 2. **Normalization Scale Mismatch (CRITICAL BUG)**
- **Input Data**: NOT normalized (raw scale)
- **Model Outputs**: NOT normalized (raw scale)
- **Ground Truth Labels**: NORMALIZED (0-1 scale)
- **Impact**: Loss computed between different scales, causing training instability
- **Evidence**:
  ```python
  # train/train.py:82-83
  # outputs = max_min_norm(outputs)  ← COMMENTED OUT!
  labels = max_min_norm(labels)      ← Only labels normalized
  ```

### 3. **Data Split Function Location**
- **Found in**: `train/dataset.py:65-72` within `__getitem__` method
- **Train Split**: Files 0-200 (first ~200 out of 207 total files)
- **Test Split**: Files 200-250 (last ~7 files)
- **Control**: Via `self.is_Train` flag and hardcoded indices

### 4. **Comment Inconsistencies**
- **Line 77**: Comment mentions "HSI,label.mask" but code only processes HSI and target
- **Line 66**: Comment says "随机选取 200 nums" but range is 0-200 (201 options)
- **Variable Naming**: Inconsistent use of `input`/`inputs`, `target`/`labels`

### 5. **Missing Input Normalization**
- Raw HSI values range from ~455-13719 (see HSI_summary_results.txt)
- Model receives unnormalized inputs with huge dynamic range
- No input preprocessing/standardization applied