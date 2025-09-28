import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class MST_dataset(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath, mask_path="/data4/zhuo-file/mask_sim.mat"):
        super(MST_dataset, self).__init__()
        self.size = 256
        self.train_set = 2560
        self.is_Train = True
        nums = 210

        # Load HSI data
        self.data = MST_dataset.load_mat_files(HSI_filepath, nums)
        # For MST, we use the same HSI data as ground truth (reconstruction task)
        self.label = self.data  # MST reconstructs HSI from compressed measurements

        # Load mask for CASSI simulation
        try:
            mat_data = sio.loadmat(mask_path)
            self.mask = mat_data['mask']
        except:
            # If mask file not found, create a random binary mask
            print(f"Warning: Could not load mask from {mask_path}, creating random mask")
            self.mask = np.random.choice([0, 1], size=(256, 256), p=[0.5, 0.5])

        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

    @staticmethod
    def load_mat_files(base_path, nums):
        """Load HSI data files"""
        data = []
        for num in range(1, nums + 1):
            if num == 15 or num == 43 or num == 109:
                continue
            print(f'Loading HSI data {num}')
            filename = os.path.join(base_path, f"{num}.mat")
            try:
                mat = sio.loadmat(filename)
                data.append(mat['extracted_data'])
            except:
                print(f"Warning: Could not load {filename}")
                continue
        return data

    def __len__(self):
        return self.train_set

    def initial_reconstruction(self, meas, mask_3d_shift):
        """Initial HSI reconstruction from 2D measurement using shift-back operation"""
        nC, step = 84, 2
        row, col = meas.shape
        out_col = row  # Reconstruct to square image
        x = np.zeros((row, out_col, nC))

        # Shift-back operation to extract spectral channels
        for i in range(nC):
            start_col = step * i
            end_col = start_col + out_col
            if end_col <= col:
                x[:, :, i] = meas[:, start_col:end_col]
            else:
                # Handle edge case where shift exceeds measurement width
                available_width = col - start_col
                x[:, :available_width, i] = meas[:, start_col:col]

        return x

    def __getitem__(self, idx):
        if self.is_Train:
            index1 = random.randint(0, min(199, len(self.data)-1))
            hsi = self.data[index1]
            target = self.label[index1]
        else:
            index1 = random.randint(min(200, len(self.data)-1), len(self.data)-1)
            hsi = self.data[index1]
            target = self.label[index1]

        shape = np.shape(hsi)

        # Random cropping
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        hsi = hsi[px:px + self.size, py:py + self.size, :]
        target = target[px:px + self.size, py:py + self.size, :]

        # Crop mask
        pxm = random.randint(0, max(0, self.mask.shape[0] - self.size))
        pym = random.randint(0, max(0, self.mask.shape[1] - self.size))
        mask_3d = self.mask_3d[pxm:pxm + self.size, pym:pym + self.size, :]

        # Data augmentation for training
        if self.is_Train:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hsi = np.rot90(hsi)
                target = np.rot90(target)
                mask_3d = np.rot90(mask_3d)

            # Random vertical flip
            if vFlip:
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()
                mask_3d = mask_3d[:, ::-1, :].copy()

            # Random horizontal flip
            if hFlip:
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()
                mask_3d = mask_3d[::-1, :, :].copy()

        # Generate CASSI measurement
        temp = mask_3d * hsi
        temp_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        temp_shift[:, 0:self.size, :] = temp
        mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:self.size, :] = mask_3d

        # Apply spectral shifting (CASSI forward model)
        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)

        # Sum to get 2D measurement
        meas = np.sum(temp_shift, axis=2)
        input_2d = meas / 84 * 0.9

        # For MST, we need to provide an initial HSI estimate
        # Use shift-back operation to create initial reconstruction
        input_hsi = self.initial_reconstruction(input_2d, mask_3d_shift)

        # Convert to tensors with proper shape [C, H, W]
        input_hsi = torch.FloatTensor(input_hsi.copy()).permute(2, 0, 1)  # [84, H, W]
        target = torch.FloatTensor(target.copy()).permute(2, 0, 1)        # [84, H, W]
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)  # [84, H, W]

        return input_hsi, target, mask_3d_shift

# Compatibility function to maintain interface with existing code
def dataset(*args, **kwargs):
    """Wrapper function for backward compatibility"""
    return MST_dataset(*args, **kwargs)