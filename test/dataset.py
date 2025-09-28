import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio

class dataset(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath, is_train=True):
        super(dataset, self).__init__()
        self.size = 256
        self.train_set = 2560
        self.test_set = 52
        self.is_Train = False
        self.nums = 250  # Load up to the 250th data
        
        # Load HSI and Labels based on training/testing flag
        if self.is_Train:
            self.data = dataset.load_mat_files(HSI_filepath, 200)  # Load first 200 for training
            self.label = dataset.load_label_files(Label_filepath, 200)
        else:
            self.data = dataset.load_mat_files(HSI_filepath, 52, start_index=204)  # Load from 200 to 250
            self.label = dataset.load_label_files(Label_filepath, 52, start_index=204)

        # Load the mask
        mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
        self.mask = mat_data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  # Third dimension replicated 84 times

    @staticmethod
    def load_mat_files(base_path, nums, start_index=1):
        data = []
        for num in range(start_index, start_index + nums):
            print(f'loading mat {num}')
            filename = os.path.join(base_path, f"{num}.mat")
            mat = sio.loadmat(filename)
            data.append(mat['extracted_data'])
        return data

    @staticmethod
    def load_label_files(base_path, nums, start_index=1):
        data = []
        for num in range(start_index, start_index + nums):
            print(f'loading label {num}')
            filename = os.path.join(base_path, f"{num}_indices.mat")
            mat = sio.loadmat(filename)
            data.append(mat['spectral_indices'])
        return data

    def __len__(self):
        return  self.test_set  # For testing, we have 50 samples

    def __getitem__(self, idx):
        if self.is_Train:  
            index1 = random.randint(0, 200) #随机选取 200
            hsi = self.data[index1] / 2600
            target = self.label[index1]
        else:
            idx1 = random.randint(0, 51)
            hsi = self.data[idx1] / 2600  # For test data, directly use idx
            target = self.label[idx1]

        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        hsi = hsi[px:px + self.size, py:py + self.size, :]
        target = target[px:px + self.size, py:py + self.size, :]

        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size, pym:pym + self.size, :]

        # Data augmentation for training only
        if self.is_Train:  
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            for j in range(rotTimes):
                hsi = np.rot90(hsi)
                target = np.rot90(target)

            for j in range(vFlip):
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()

            for j in range(hFlip):
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        # Generate measurement frame
        temp = mask_3d * hsi
        temp_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        temp_shift[:, 0:self.size, :] = temp
        mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:self.size, :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)

        input = meas / 84 * 0.9

        # Simulate uncertainty by introducing noise
        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)
        input = torch.FloatTensor(input.copy())
        target = torch.FloatTensor(target.copy()).permute(2, 0, 1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)

        return input, target
