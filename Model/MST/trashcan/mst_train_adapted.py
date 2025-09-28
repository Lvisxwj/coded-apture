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
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import clip_grad_norm_

# Import loss functions from parent directory
sys.path.append("../../train")
from loss import CharbonnierLoss, SSIM

# Data paths
hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"  # Same as input for MST reconstruction

# Load dataset
Dataset = MST_dataset(hsi_filename, label_filename)

# Model setup
model = MST(dim=84, stage=3, num_blocks=[2,2,2])
Model_name = "MST"
file_path = "/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/"
log_file = f"{file_path}parameter/ours/training_log_{Model_name}_lr4-4_bs4_Epoch300_2560_hybridLoss.txt"

# GPU setup
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
learning_rate = 0.0004
milestones = [100, 200, 250]
gamma = 0.5
batch_size = 4  # Reduced batch size due to MST's memory requirements
Max_epoch = 300
train_set = 2560

# Loss functions
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_Charbonnier_loss = 0.0
    epoch_L1_loss = 0.0
    epoch_SSIM_loss = 0.0

    start_time = time.time()

    for i, (inputs, labels, masks) in enumerate(train_loader):
        # Move to device
        inputs = inputs.to(device)  # [bs, 84, H, W] - initial HSI estimate
        labels = labels.to(device)  # [bs, 84, H, W] - ground truth HSI
        masks = masks.to(device)    # [bs, 84, H, W] - shifted masks

        # Forward pass
        outputs = model(inputs, masks)

        # Apply normalization - controlled in training script as requested
        # Normalize both outputs and labels for consistent loss computation
        outputs_norm = HSI_max_min_norm(outputs)
        labels_norm = HSI_max_min_norm(labels)

        # Compute losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs_norm, labels_norm)
        L1_loss = loss_L1(outputs_norm, labels_norm)
        SSIM_loss = loss_SSIM(outputs_norm, labels_norm)
        loss = Charbonnier_loss + L1_loss + SSIM_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_Charbonnier_loss += Charbonnier_loss.item()
        epoch_L1_loss += L1_loss.item()
        epoch_SSIM_loss += SSIM_loss.item()

        # Log progress
        if i % 50 == 0:
            log_str = f'{epoch:4d} {i:4d} / {train_set // batch_size:4d} loss = {loss.item():.10f} time = {datetime.datetime.now()}\n'
            print(log_str.strip())

    elapsed_time = time.time() - start_time
    avg_loss = epoch_loss / (train_set // batch_size)
    avg_Charbonnier = epoch_Charbonnier_loss / (train_set // batch_size)
    avg_L1 = epoch_L1_loss / (train_set // batch_size)
    avg_SSIM = epoch_SSIM_loss / (train_set // batch_size)

    return avg_loss, avg_Charbonnier, avg_L1, avg_SSIM, elapsed_time

if __name__ == "__main__":
    print(f'Learning rate: {learning_rate}, Max_epoch: {Max_epoch}')
    print(f'Model: {Model_name}, Batch size: {batch_size}')
    print(f'Device: {device}')

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    Whole_Loss = []

    with open(log_file, "w") as f:
        f.write(f"Training log for {Model_name}\n")
        f.write(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Max epochs: {Max_epoch}\n")
        f.write(f"Training started at: {datetime.datetime.now()}\n\n")

        for epoch in range(1, Max_epoch + 1):
            print(f"\\n=== Epoch {epoch}/{Max_epoch} ===")

            # Create data loader for this epoch
            train_loader = DataLoader(Dataset, num_workers=4, batch_size=batch_size, shuffle=True)

            # Train one epoch
            avg_loss, avg_Charbonnier, avg_L1, avg_SSIM, elapsed_time = train_epoch(
                model, train_loader, optimizer, device, epoch
            )

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log epoch results
            epoch_log_str = f'Epoch {epoch:4d}: loss = {avg_loss:.10f}, time = {elapsed_time:.2f}s, lr = {current_lr:.6f}\n'
            epoch_log_str += f'  Charbonnier: {avg_Charbonnier:.10f}, L1: {avg_L1:.10f}, SSIM: {avg_SSIM:.10f}\n'

            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str.strip())

            Whole_Loss.append(avg_loss)

            # Save model checkpoint
            if epoch % 50 == 0 or epoch == Max_epoch:
                model_save_path = f"{file_path}parameter/ours/{Model_name}_{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{epoch}_{train_set}_hybridLoss.pth"
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model, model_save_path)
                print(f"Model saved at epoch {epoch}: {model_save_path}")

        f.write(f"\\nTraining completed at: {datetime.datetime.now()}\\n")

    # Save final model
    final_model_path = f"{file_path}parameter/ours/{Model_name}/{Model_name}_lr{learning_rate:.0e}_bs{batch_size}_Epoch{Max_epoch}_{train_set}_hybridLoss.pth"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model, final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(Whole_Loss) + 1), Whole_Loss, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{Model_name} Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_plot_path = f"{file_path}parameter/ours/{Model_name}_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Loss curve saved: {loss_plot_path}")
    print("Training completed successfully!")