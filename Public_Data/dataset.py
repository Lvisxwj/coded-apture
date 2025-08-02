import torch.utils.data as tud
import random
import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import h5py

class dataset(tud.Dataset):
    def __init__(self, HSI_filepath, Label_filepath):
        super(dataset, self).__init__()
        self.size = 256
        self.train_set = 2560 #和train.py中的train_set一致
        self.is_Train = True

        # 载入HSI数据
        self.data = dataset.load_mat_files(HSI_filepath)
        # 载入Label数据
        self.label = dataset.load_label_files(Label_filepath)

        # 使用loadmat加载MAT文件
        mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
        self.mask = mat_data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  # 第三维度复制84次

    # 载入单张HSI图像（公共数据集）
    def load_mat_files(base_path):
        data = []
        print('loading mat file')
        with h5py.File(base_path, 'r') as mat:
            extracted_data = mat['extracted_data'][:]
            extracted_data = extracted_data.transpose(2, 1, 0)
            data.append(extracted_data)
        return data
    
    # 载入单张Label（公共数据集）
    def load_label_files(base_path):
        data = []
        print('loading label file')
        with h5py.File(base_path, 'r') as mat:  
            extracted_data = mat['spectral_indices'][:]
            extracted_data = extracted_data.transpose(2, 1, 0)
            data.append(extracted_data)
        return data

    def __len__(self):
        return self.train_set

    def __getitem__(self, idx):
        # 获取HSI和Label数据
        hsi = self.data[0] / 2600
        target = self.label[0]
        
        shape1 = np.shape(hsi)
        # print(shape)
        cut_index = int(shape1[1] * 0.8)

        # 如果是训练集，截取左边80%
        if self.is_Train:
            hsi = hsi[:, :cut_index, :]
            target = target[:, :cut_index, :]
        else:
            # 如果是测试集，截取右边20%
            hsi = hsi[:, cut_index:, :]
            target = target[:, cut_index:, :]
        # print(np.shape(hsi),np.shape(target))
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
        temp_shift[:, 0:self.size, :] = temp
        mask_3d_shift = np.zeros((self.size, self.size + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:self.size, :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)

        input = meas / 84 * 0.9

        #相当于模拟现实世界的不确定性，引入噪声；
        QE, bit = 0.4, 2048
        input = np.random.binomial((input * bit / QE).astype(int), QE)
        input = np.float32(input) / np.float32(bit)
        input = torch.FloatTensor(input.copy())
        target = torch.FloatTensor(target.copy()).permute(2,0,1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)


        return input, target
