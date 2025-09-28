from torch.autograd import Variable
import sys
sys.path.append("/data4/zhuo-file/fyc_file/ATTU_Test/")
import torch
from torch.utils.data import DataLoader
from Model import UNet
from Model import *
import datetime
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import numpy as np
from loss import *
from torch.nn.utils import clip_grad_norm_
from Norm import max_min_norm
import scipy.io as sio
import random

hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/cor_r_label/"

# 自定义测试数据集，只测试特定文件210号，不做数据增强
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_path, label_path, file_index=20, start_H=0, start_W=0):
        super(TestDataset, self).__init__()
        self.file_index = file_index
        self.hsi_path = hsi_path
        self.label_path = label_path
        self.start_H = start_H
        self.start_W = start_W
        
        # 加载HSI数据
        mat = sio.loadmat(os.path.join(self.hsi_path, f"{self.file_index}.mat"))
        self.hsi = mat['extracted_data'] / 2600.0  # 数据归一化
        # print('hsi.shape', self.hsi.shape)
        shape = self.hsi.shape
        
        # 加载label数据
        mat_label = sio.loadmat(os.path.join(self.label_path, f"{self.file_index}_indices.mat"))
        self.label = mat_label['spectral_indices']
        # print('label.shape', self.label.shape)

        # 加载mask
        mat_data = sio.loadmat(r"/data4/zhuo-file/mask_sim.mat")
        mask = mat_data['mask']  # 原始 (256, 256)
        # print(f"Original Shape of mask: {mask.shape}")

        # 将 mask 的长宽各复制为原来的 5 倍
        self.mask = np.repeat(np.repeat(mask, 5, axis=0), 5, axis=1)
        # print(f"Resized Shape of mask: {self.mask.shape}")

        # 沿第三维度复制 84 次
        mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))
        
        self.mask_3d = mask_3d[:1072, :1004, :]
        # print(f"Shape of mask_3d after cropping: {self.mask_3d.shape}")

        # 新增的500x500裁剪 
        self.hsi = self.hsi[self.start_H:self.start_H+256, self.start_W:self.start_W+256, :]
        self.label = self.label[self.start_H:self.start_H+256, self.start_W:self.start_W+256, :]
        self.mask_3d = self.mask_3d[self.start_H:self.start_H+256, self.start_W:self.start_W+256, :]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        hsi = self.hsi
        target = self.label
        mask_3d = self.mask_3d
        shape = hsi.shape

        # 生成测量帧
        temp = mask_3d * hsi
        temp_shift = np.zeros((shape[0], shape[1] + (84 - 1) * 2, 84))
        temp_shift[:, 0:shape[1], :] = temp
        mask_3d_shift = np.zeros((shape[0], shape[1] + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:shape[1], :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2*t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2*t, axis=1)
        meas = np.sum(temp_shift, axis=2)

        input_data = meas / 84 * 0.9
        QE, bit = 0.4, 2048
        input_data = np.random.binomial((input_data * bit / QE).astype(int), QE)
        input_data = np.float32(input_data) / np.float32(bit)
        input_data = torch.FloatTensor(input_data.copy())
        target = torch.FloatTensor(target.copy()).permute(2, 0, 1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        # print('input.shape', input_data.shape)

        return input_data, target


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/parameter/normal/ATTU_1e-3/CTRM_UNetW4C_lr1-3_bs1_Epoch300_2560_hybridLoss.pth", map_location="cuda:0") 
model.to(device)

loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

best_L1_loss = float('inf')
best_start_H = 0
best_start_W = 0

for _ in range(100):
    print('Epoch:',_)
    start_H = random.randint(0, 700)
    start_W = random.randint(0, 700)
    
    test_dataset = TestDataset(hsi_filename, label_filename, file_index=210, start_H=start_H, start_W=start_W)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print('outputs.shape', outputs.shape)
            
            # Normalize outputs and labels
            outputs = max_min_norm(outputs)
            labels = max_min_norm(labels)

            # Calculate L1 loss
            L1_loss = loss_L1(outputs, labels)
            print (f"Start_H: {start_H}, Start_W: {start_W}, L1 Loss: {L1_loss:.4f}")

            # Update best L1 loss and corresponding start_H, start_W
            if L1_loss < best_L1_loss:
                best_L1_loss = L1_loss
                best_start_H = start_H
                best_start_W = start_W

print(f"Best Start_H: {best_start_H}, Best Start_W: {best_start_W}, Best L1 Loss: {best_L1_loss:.4f}")

# 使用最佳的 start_H 和 start_W 再次运行模型并保存结果
test_dataset = TestDataset(hsi_filename, label_filename, file_index=210, start_H=best_start_H, start_W=best_start_W)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # print('outputs.shape', outputs.shape)
        
        # 将输出保存为mat文件
        outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
        save_path_output = "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/Sample/210/210_best_result.mat"
        sio.savemat(save_path_output, {"output": outputs_np})
        print(f"Saved best result to {save_path_output}")
        
        # 将label也保存下来，以不同的文件名
        labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
        save_path_label = "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/Sample/210/210_best_label.mat"
        sio.savemat(save_path_label, {"label": labels_np})
        print(f"Saved best label to {save_path_label}")
        
        # Normalize outputs and labels
        outputs = max_min_norm(outputs)
        labels = max_min_norm(labels)

        # Calculate losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
        L1_loss = loss_L1(outputs, labels)
        SSIM_loss = loss_SSIM(outputs, labels)
        loss = Charbonnier_loss + L1_loss + SSIM_loss
        
        print (f"Charbonnier Loss: {Charbonnier_loss:.4f}, L1 Loss: {L1_loss:.4f}, SSIM Loss: {SSIM_loss:.4f}, Total Loss: {loss:.4f}")