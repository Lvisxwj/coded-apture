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
    def __init__(self, HSI_filepath, Label_filepath):
        super(dataset, self).__init__()
        self.size = 256
        self.train_set = 2560 #和train.py中的train_set一致
        self.is_Train = True
        nums = 210 #加载数据的个数 255

        # 载入HSI
        self.data = dataset.load_mat_files(HSI_filepath, nums)
        # 载入Label
        self.label = dataset.load_label_files(Label_filepath, nums)

        # 使用loadmat加载MAT文件
        mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
        self.mask = mat_data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))  #第三维度复制84次
    
    #按照文件名的排序读取数据并按顺序放到data中
    def load_mat_files(base_path, nums):
        data = []
        for num in range(1, nums + 1):
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
    def load_label_files(base_path, nums):
        data = []
        for num in range(1, nums + 1):
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
        return self.train_set

    def __getitem__(self, idx):
        if self.is_Train:  
            index1 = random.randint(0, 200) #随机选取 200 nums
            hsi = self.data[index1]
            target = self.label[index1]
        else:
            index1 = random.randint(200, 250) #随机选取 200-250
            hsi = self.data[index1]
            target = self.label[index1]

        shape = np.shape(hsi) #读取hsi的大小
        # print(shape)

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

        #相当于模拟现实世界的不确定性，引入噪声；这部分代码没用了，应该省略掉
        # QE, bit = 0.4, 2048
        # input = np.random.binomial((input * bit / QE).astype(int), QE)
        # input = np.float32(input) / np.float32(bit)
        input = torch.FloatTensor(input.copy())
        hsi = torch.FloatTensor(hsi.copy()).permute(2,0,1)
        target = torch.FloatTensor(target.copy()).permute(2,0,1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2,0,1)


        return input, target






 # def load_label_files(base_path, nums):
    #     all_data = []

    #     for num in range(1, nums + 1):
    #         if num == 15:
    #             continue
    #         print(f'loading label {num}')
    #         # 构建 mat 文件路径
    #         filename = os.path.join(base_path, f"{num}_indices.mat")
            
    #         if os.path.exists(filename):
    #             # 加载 mat 数据
    #             mat_data = sio.loadmat(filename)

    #             # 假设你的键名是 'spectral_indices'
    #             spectral_indices = mat_data['spectral_indices']

    #             # 如果 spectral_indices 是一个 numpy 数组
    #             if isinstance(spectral_indices, np.ndarray):
    #                 # 将第一个维度（key）展开
    #                 keys = spectral_indices.dtype.names
    #                 indices = [spectral_indices[key][0, 0] for key in keys]

    #                 # 将 indices 按第三维度堆叠起来
    #                 stacked_indices = np.stack(indices, axis=2)

    #                 # 将 stacked_indices 添加到总数据列表中
    #                 all_data.append(stacked_indices)

    #     # 将列表中的数据转换为 NumPy 数组
    #     #all_data = np.array(all_data)

    #     return all_data
        
