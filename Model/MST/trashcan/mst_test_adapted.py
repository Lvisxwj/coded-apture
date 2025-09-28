from torch.autograd import Variable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from MST import MST
from mst_dataset_adapted import MST_dataset
from mst_norm import HSI_max_min_norm, HSI_reverse_max_min_norm, prepare_mask_for_mst
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Import loss functions from parent directory
sys.path.append("../../train")
sys.path.append("../../test")
from loss import CharbonnierLoss, SSIM

# Data paths
hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"  # Same as input for MST reconstruction

# Load dataset
Dataset = MST_dataset(hsi_filename, label_filename)
Dataset.is_Train = False  # Set to test mode

# GPU setup
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load trained model
model_path = r"/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/parameter/ours/MST/MST_lr4-4_bs4_Epoch300_2560_hybridLoss.pth"
try:
    model = torch.load(model_path, map_location=device)
    print(f"Loaded model from: {model_path}")
except:
    print(f"Could not load model from {model_path}, initializing new model")
    model = MST(dim=84, stage=3, num_blocks=[2,2,2])

model.to(device)

# Test parameters
batch_size = 1
Max_epoch = 1
test_set = 1200

# Loss functions
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_sam(img1, img2):
    """Calculate Spectral Angle Mapper (SAM) between two hyperspectral images"""
    # Flatten spatial dimensions
    img1_flat = img1.view(img1.shape[0], img1.shape[1], -1)  # [bs, channels, H*W]
    img2_flat = img2.view(img2.shape[0], img2.shape[1], -1)

    # Calculate SAM for each pixel
    dot_product = torch.sum(img1_flat * img2_flat, dim=1)  # [bs, H*W]
    norm1 = torch.norm(img1_flat, dim=1)  # [bs, H*W]
    norm2 = torch.norm(img2_flat, dim=1)  # [bs, H*W]

    # Avoid division by zero
    norm_product = norm1 * norm2
    norm_product = torch.clamp(norm_product, min=1e-8)

    cos_angle = dot_product / norm_product
    cos_angle = torch.clamp(cos_angle, -1, 1)  # Ensure valid range for acos

    sam_map = torch.acos(cos_angle)  # [bs, H*W]
    sam_mean = torch.mean(sam_map)

    return sam_mean * 180 / np.pi  # Convert to degrees

def test_model(model, test_loader, device, save_results=True):
    """Test the model and calculate metrics"""
    model.eval()

    epoch_loss = 0.0
    epoch_Charbonnier_loss = 0.0
    epoch_L1_loss = 0.0
    epoch_SSIM_loss = 0.0

    all_psnr = []
    all_sam = []

    all_outputs = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for i, (inputs, labels, masks) in enumerate(test_loader):
            # Move to device
            inputs = inputs.to(device)  # [bs, 84, H, W] - initial HSI estimate
            labels = labels.to(device)  # [bs, 84, H, W] - ground truth HSI
            masks = masks.to(device)    # [bs, 84, H, W] - shifted masks

            # Forward pass
            outputs = model(inputs, masks)

            # Apply normalization for loss computation
            outputs_norm = HSI_max_min_norm(outputs)
            labels_norm = HSI_max_min_norm(labels)

            # Calculate losses
            Charbonnier_loss = loss_CharbonnierLoss(outputs_norm, labels_norm)
            L1_loss = loss_L1(outputs_norm, labels_norm)
            SSIM_loss = loss_SSIM(outputs_norm, labels_norm)
            loss = Charbonnier_loss + L1_loss + SSIM_loss

            # Accumulate losses
            epoch_loss += loss.item()
            epoch_Charbonnier_loss += Charbonnier_loss.item()
            epoch_L1_loss += L1_loss.item()
            epoch_SSIM_loss += SSIM_loss.item()

            # Calculate additional metrics
            psnr = calculate_psnr(outputs_norm, labels_norm)
            sam = calculate_sam(outputs, labels)  # Use original scale for SAM

            all_psnr.append(psnr.item())
            all_sam.append(sam.item())

            # Store results for analysis
            if save_results:
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            # Log progress
            if i % 100 == 0:
                log_str = f'{i:4d} / {test_set // batch_size:4d} '
                log_str += f'loss = {epoch_loss / (i + 1):.10f} '
                log_str += f'PSNR = {psnr:.4f} '
                log_str += f'SAM = {sam:.4f} '
                log_str += f'time = {datetime.datetime.now()}\n'
                print(log_str.strip())

    elapsed_time = time.time() - start_time

    # Calculate average metrics
    avg_loss = epoch_loss / (test_set // batch_size)
    avg_Charbonnier = epoch_Charbonnier_loss / (test_set // batch_size)
    avg_L1 = epoch_L1_loss / (test_set // batch_size)
    avg_SSIM = epoch_SSIM_loss / (test_set // batch_size)
    avg_psnr = np.mean(all_psnr)
    avg_sam = np.mean(all_sam)

    results = {
        'avg_loss': avg_loss,
        'avg_Charbonnier': avg_Charbonnier,
        'avg_L1': avg_L1,
        'avg_SSIM': avg_SSIM,
        'avg_psnr': avg_psnr,
        'avg_sam': avg_sam,
        'elapsed_time': elapsed_time,
        'all_outputs': all_outputs if save_results else None,
        'all_labels': all_labels if save_results else None
    }

    return results

if __name__ == "__main__":
    log_file = r"./MST_test_log.txt"
    results_dir = r"./MST_test_results/"

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    print(f'Testing MST model on {device}')
    print(f'Test set size: {test_set}')

    with open(log_file, "w") as f:
        f.write(f"MST Test Results\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test started at: {datetime.datetime.now()}\n\n")

        for epoch in range(1, Max_epoch + 1):
            print(f"\\n=== Test Epoch {epoch}/{Max_epoch} ===")

            # Create test data loader
            test_loader = DataLoader(Dataset, num_workers=4, batch_size=batch_size, shuffle=False)

            # Test the model
            results = test_model(model, test_loader, device, save_results=True)

            # Log results
            log_str = f'Test Results:\n'
            log_str += f'  Total Loss: {results["avg_loss"]:.10f}\n'
            log_str += f'  Charbonnier Loss: {results["avg_Charbonnier"]:.10f}\n'
            log_str += f'  L1 Loss: {results["avg_L1"]:.10f}\n'
            log_str += f'  SSIM Loss: {results["avg_SSIM"]:.10f}\n'
            log_str += f'  PSNR: {results["avg_psnr"]:.4f} dB\n'
            log_str += f'  SAM: {results["avg_sam"]:.4f} degrees\n'
            log_str += f'  Test Time: {results["elapsed_time"]:.2f} seconds\n'

            f.write(log_str)
            f.flush()
            print(log_str)

            # Save detailed results
            if results["all_outputs"] is not None:
                # Save some sample results
                for idx in range(min(5, len(results["all_outputs"]))):
                    output_sample = results["all_outputs"][idx][0]  # First item in batch
                    label_sample = results["all_labels"][idx][0]

                    # Save as .mat files
                    sio.savemat(
                        os.path.join(results_dir, f'sample_{idx}_output.mat'),
                        {'output': output_sample, 'label': label_sample}
                    )

                    # Create RGB visualization (using bands 30, 20, 10 for RGB)
                    if output_sample.shape[0] >= 31:
                        rgb_output = np.stack([
                            output_sample[30],  # Red
                            output_sample[20],  # Green
                            output_sample[10]   # Blue
                        ], axis=2)

                        rgb_label = np.stack([
                            label_sample[30],
                            label_sample[20],
                            label_sample[10]
                        ], axis=2)

                        # Normalize for visualization
                        rgb_output = (rgb_output - rgb_output.min()) / (rgb_output.max() - rgb_output.min())
                        rgb_label = (rgb_label - rgb_label.min()) / (rgb_label.max() - rgb_label.min())

                        # Save RGB images
                        plt.figure(figsize=(12, 6))
                        plt.subplot(1, 2, 1)
                        plt.imshow(rgb_output)
                        plt.title('MST Output (RGB)')
                        plt.axis('off')

                        plt.subplot(1, 2, 2)
                        plt.imshow(rgb_label)
                        plt.title('Ground Truth (RGB)')
                        plt.axis('off')

                        plt.tight_layout()
                        plt.savefig(os.path.join(results_dir, f'sample_{idx}_rgb_comparison.png'),
                                  dpi=300, bbox_inches='tight')
                        plt.close()

        f.write(f"\\nTesting completed at: {datetime.datetime.now()}\\n")

    print(f"Test results saved to: {results_dir}")
    print(f"Test log saved to: {log_file}")
    print("Testing completed successfully!")