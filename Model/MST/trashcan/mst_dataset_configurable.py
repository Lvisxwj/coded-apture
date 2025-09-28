import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import pickle

class MST_dataset_configurable(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath, mask_path="/data4/zhuo-file/mask_sim.mat",
                 train_split=0.9, train_files=None, test_files=None, total_files=210):
        """
        Configurable MST dataset with controllable data splits

        Args:
            HSI_filepath: Path to HSI data
            Label_filepath: Path to label data
            mask_path: Path to mask file
            train_split: Fraction for training (0.0-1.0), ignored if train_files/test_files provided
            train_files: Specific file indices for training (e.g., [0, 50, 100])
            test_files: Specific file indices for testing (e.g., [200, 205])
            total_files: Total number of files to load (default 210)
        """
        super(MST_dataset_configurable, self).__init__()
        self.size = 256
        self.train_set = 2560
        self.is_Train = True
        self.total_files = total_files

        # Load HSI data
        self.data = MST_dataset_configurable.load_mat_files(HSI_filepath, self.total_files)
        # For MST, we use the same HSI data as ground truth (reconstruction task)
        self.label = self.data  # MST reconstructs HSI from compressed measurements

        # Configure data splits
        self._configure_data_splits(train_split, train_files, test_files)

        # Load mask for CASSI simulation
        try:
            mat_data = sio.loadmat(mask_path)
            self.mask = mat_data['mask']
        except:
            print(f"Warning: Could not load mask from {mask_path}, creating random mask")
            self.mask = np.random.choice([0, 1], size=(256, 256), p=[0.5, 0.5])

        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

    def _configure_data_splits(self, train_split, train_files, test_files):
        """Configure training and testing file indices"""
        total_available = len(self.data)

        if train_files is not None and test_files is not None:
            # Use explicitly provided file lists
            self.train_indices = train_files
            self.test_indices = test_files
            print(f"Using explicit file splits: {len(train_files)} train, {len(test_files)} test")
        elif train_files is not None:
            # Use provided training files, rest for testing
            self.train_indices = train_files
            self.test_indices = [i for i in range(total_available) if i not in train_files]
            print(f"Using explicit train files: {len(self.train_indices)} train, {len(self.test_indices)} test")
        else:
            # Use train_split ratio
            split_point = int(total_available * train_split)
            self.train_indices = list(range(split_point))
            self.test_indices = list(range(split_point, total_available))
            print(f"Using {train_split:.1%} split: {len(self.train_indices)} train, {len(self.test_indices)} test")

        print(f"Train indices: {self.train_indices[:5]}...{self.train_indices[-5:] if len(self.train_indices) > 5 else self.train_indices}")
        print(f"Test indices: {self.test_indices[:5]}...{self.test_indices[-5:] if len(self.test_indices) > 5 else self.test_indices}")

    def set_mode(self, is_train=True):
        """Set dataset to training or testing mode"""
        self.is_Train = is_train
        print(f"Dataset mode set to: {'Training' if is_train else 'Testing'}")

    def get_split_info(self):
        """Return information about current data split"""
        return {
            'total_files': len(self.data),
            'train_files': len(self.train_indices),
            'test_files': len(self.test_indices),
            'train_indices': self.train_indices,
            'test_indices': self.test_indices,
            'current_mode': 'train' if self.is_Train else 'test'
        }

    @staticmethod
    def load_mat_files(base_path, nums):
        """Load HSI data files"""
        data = []
        corrupted_files = [15, 43, 109]

        for num in range(1, nums + 1):
            if num in corrupted_files:
                continue
            print(f'Loading HSI data {num}')
            filename = os.path.join(base_path, f"{num}.mat")
            try:
                mat = sio.loadmat(filename)
                data.append(mat['extracted_data'])
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                continue

        print(f"Successfully loaded {len(data)} files")
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
            if len(self.train_indices) == 0:
                raise ValueError("No training files available!")
            index1 = random.choice(self.train_indices)
        else:
            if len(self.test_indices) == 0:
                raise ValueError("No test files available!")
            index1 = random.choice(self.test_indices)

        hsi = self.data[index1]
        target = self.label[index1]

        shape = np.shape(hsi)

        # Random cropping
        px = random.randint(0, shape[0] - self.size) if shape[0] > self.size else 0
        py = random.randint(0, shape[1] - self.size) if shape[1] > self.size else 0
        hsi = hsi[px:px + self.size, py:py + self.size, :]
        target = target[px:px + self.size, py:py + self.size, :]

        # Crop mask
        mask_h, mask_w = self.mask.shape
        pxm = random.randint(0, max(0, mask_h - self.size))
        pym = random.randint(0, max(0, mask_w - self.size))
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

# Convenience functions for common configurations
def create_mst_dataset_small(hsi_path, label_path, train_size=50, test_size=10):
    """Create a small dataset for quick experiments"""
    total_files = train_size + test_size
    train_files = list(range(train_size))
    test_files = list(range(train_size, total_files))

    return MST_dataset_configurable(
        hsi_path, label_path,
        train_files=train_files,
        test_files=test_files,
        total_files=total_files
    )

def create_mst_dataset_custom_split(hsi_path, label_path, train_ratio=0.8, total_files=100):
    """Create dataset with custom split ratio and file count"""
    return MST_dataset_configurable(
        hsi_path, label_path,
        train_split=train_ratio,
        total_files=total_files
    )

# Compatibility function to maintain interface with existing code
def dataset(*args, **kwargs):
    """Wrapper function for backward compatibility"""
    return MST_dataset_configurable(*args, **kwargs)