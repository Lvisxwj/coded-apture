from torch.autograd import Variable
import sys
sys.path.append("/root/Code/ATTU_Test/")
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
from Norm import reverse_max_min_norm
import scipy.io as sio
from PIL import Image

index = 220
hsi_filename = r"/root/autodl-tmp/private_data/5nm_input/"
label_filename = r"/root/autodl-tmp/private_data/cor_r_label/"

def minmax_norm(channel):
    c_min = channel.min()
    c_max = channel.max()
    if abs(c_max - c_min) < 1e-12:
        return np.zeros_like(channel)  # 避免除以0
    return (channel - c_min) / (c_max - c_min)

# 自定义测试数据集，只测试特定文件210号，不做数据增强
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_path, label_path, file_index):
        super(TestDataset, self).__init__()
        self.file_index = file_index
        self.hsi_path = hsi_path
        self.label_path = label_path
        
        # 加载HSI数据
        mat = sio.loadmat(os.path.join(self.hsi_path, f"{self.file_index}.mat"))
        self.hsi = mat['extracted_data'] / 2600.0  # 数据归一化
        print('hsi.shape', self.hsi.shape)
        shape = self.hsi.shape
        
        # 加载label数据
        mat_label = sio.loadmat(os.path.join(self.label_path, f"{self.file_index}_indices.mat"))
        self.label = mat_label['spectral_indices']
        print('label.shape', self.label.shape)

        # 加载mask
        mat_data = sio.loadmat(r"/root/autodl-tmp/mask_sim.mat")
        mask = mat_data['mask']  # 原始 (256, 256)
        print(f"Original Shape of mask: {mask.shape}")

        # 将 mask 的长宽各复制为原来的 5 倍
        self.mask = np.repeat(np.repeat(mask, 5, axis=0), 5, axis=1)
        print(f"Resized Shape of mask: {self.mask.shape}")

        # 沿第三维度复制 84 次
        mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))
        
        self.mask_3d = mask_3d[:1143, :1004, :]
        print(f"Shape of mask_3d after cropping: {self.mask_3d.shape}")

        start_H = 250 # TODO
        start_W = 100
        # 新增的500x500裁剪 
        self.hsi = self.hsi[start_H:start_H+256, start_W:start_W+256, :]
        self.label = self.label[start_H:start_H+256, start_W:start_W+256, :]
        self.mask_3d = self.mask_3d[start_H:start_H+256, start_W:start_W+256, :]

        # 保存为RGB图像
        input_sample = self.hsi
        print('input_sample', input_sample.shape)
        r = input_sample[:, :, 44]
        g = input_sample[:, :, 23]
        b = input_sample[:, :, 1]
        # 分别归一化
        r_norm = minmax_norm(r)
        g_norm = minmax_norm(g)
        b_norm = minmax_norm(b)
        rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
        rgb_image = (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_image, mode='RGB')
        img.save(f"/root/Code/Sample/ours220/RGB(250,100).png")

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
        print('input.shape', input_data.shape)

        return input_data, target


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "/root/Code/parameter/ours/UNet/UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth", map_location="cuda:0") 
model.to(device)

loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

test_dataset = TestDataset(hsi_filename, label_filename, file_index=index)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print('outputs.shape', outputs.shape)
        
        # 将label也保存下来，以不同的文件名
        labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
        save_path_label = "/root/Code/Sample/ours220/UNet/label.mat"
        sio.savemat(save_path_label, {"label": labels_np})
        print(f"Saved label to {save_path_label}")
        
        # Normalize outputs and labels
        labels = max_min_norm(labels)

        # Calculate losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
        L1_loss = loss_L1(outputs, labels)
        SSIM_loss = loss_SSIM(outputs, labels)
        loss = Charbonnier_loss + L1_loss + SSIM_loss
         
        print (f"Charbonnier Loss: {Charbonnier_loss:.4f}, L1 Loss: {L1_loss:.4f}, SSIM Loss: {SSIM_loss:.4f}, Total Loss: {loss:.4f}")
        
        outputs = reverse_max_min_norm(outputs)
        # 将输出保存为mat文件
        outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
        save_path_output = "/root/Code/Sample/ours220/UNet/result.mat"
        sio.savemat(save_path_output, {"output": outputs_np})
        print(f"Saved result to {save_path_output}")

        
    '''
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print('outputs.shape', outputs.shape)
        
        # 将输出保存为mat文件
        outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
        save_path_output = "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/Sample/new210/210_result.mat"
        sio.savemat(save_path_output, {"output": outputs_np})
        print(f"Saved result to {save_path_output}")
        
        # 将label也保存下来，以不同的文件名
        labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
        save_path_label = "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/Sample/new210/210_label.mat"
        sio.savemat(save_path_label, {"label": labels_np})
        print(f"Saved label to {save_path_label}")
        
        # Normalize outputs and labels
        outputs = max_min_norm(outputs)
        labels = max_min_norm(labels)

        # Calculate losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
        L1_loss = loss_L1(outputs, labels)
        SSIM_loss = loss_SSIM(outputs, labels)
        loss = Charbonnier_loss + L1_loss + SSIM_loss
        
        print (f"Charbonnier Loss: {Charbonnier_loss:.4f}, L1 Loss: {L1_loss:.4f}, SSIM Loss: {SSIM_loss:.4f}, Total Loss: {loss:.4f}")
    '''