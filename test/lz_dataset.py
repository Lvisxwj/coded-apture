import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import pickle


class dataset(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath, is_train=True):
        super(dataset, self).__init__()
        self.size = 256
        self.train_set = 2560
        self.test_set = 2560 
        self.is_Train = False
        self.nums = 255  # Load up to the 250th data
        
        
        
        # Load HSI and Labels based on training/testing flag
        if self.is_Train:
            self.data = dataset.load_mat_files(HSI_filepath, 204)  # Load first 200 for training
            self.label = dataset.load_label_files(Label_filepath, 204)
        else:
            self.data = dataset.load_mat_files(HSI_filepath, 52, start_index=204)  # Load from 200 to 250
            self.label = dataset.load_label_files(Label_filepath, 52, start_index=204)
        
        '''
        self.data = dataset.load_mat_files(HSI_filepath, 255)  # Load first 200 for training
        self.label = dataset.load_label_files(Label_filepath, 255)        
        '''

        # Load the mask
        mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
        self.mask = mat_data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  # Third dimension replicated 84 times

    #按照文件名的排序读取数据并按顺序放到data中
    def load_mat_files(base_path, nums, start_index):
        data = []
        for num in range(start_index, nums + start_index):
            if num == 15 or num == 43 or num == 109:
                continue
            print(f'loading mat {num}')
            # 构建 mat 文件路径
            filename = os.path.join(base_path, f"{num}.mat")
            # 加载 mat 数据
            mat = sio.loadmat(filename)
            data.append(mat['extracted_data'])
            
            
        return data
    
     #按照文件名的排序读取数据并按顺序放到data中
    def load_label_files(base_path, nums, start_index):
        data = []
        for num in range(start_index, nums + start_index):
            if num == 15 or num == 43 or num == 109:
                continue
            print(f'loading label {num}')
            # 构建 mat 文件路径
            filename = os.path.join(base_path, f"{num}_indices.mat")
            # 加载 mat 数据
            mat = sio.loadmat(filename)
            data.append(mat['spectral_indices'])
            
        return data
        

    def __len__(self):
        return self.test_set

    def __getitem__(self, idx):
        if self.is_Train:
            index1 = random.randint(0, 200)  # 随机选取     //TODO 后期看看能不能增加
            hsi = self.data[index1] / 2600
            target = self.label[index1]
        else:
            idx = random.randint(0,51)
            hsi = self.data[idx] / 2600  # 从182顺序到221共40个数据，idx 0-39
            target = self.label[idx]

        shape = np.shape(hsi)

        # 对HSI,label.mask进行剪切，使其变成 self.size × self.size
        # 通过随机数，随机剪切其中的一部分
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        hsi = hsi[px:px + self.size:1, py:py + self.size:1, :]
        target = target[px:px + self.size:1, py:py + self.size:1, :]

        # 对mask进行剪切，使其变成 self.size × self.size
        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        # 进行翻转
        if self.is_Train:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hsi = np.rot90(hsi)
                target = np.rot90(target)

            # Random vertical Flip
            for j in range(vFlip):
                hsi = hsi[:, ::-1, :].copy()
                target = target[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hsi = hsi[::-1, :, :].copy()
                target = target[::-1, :, :].copy()

        # 生成测量帧
        temp = mask_3d * hsi
        temp_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        temp_shift[:, 0:self.size, :] = temp  # 64, 64, 84
        mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:self.size, :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)

        input = meas / 84 * 0.9

        # 相当于模拟现实世界的不确定性，引入噪声；
        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)
        input = torch.FloatTensor(input.copy())
        target = torch.FloatTensor(target.copy()).permute(2, 0, 1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)

        return input, target

