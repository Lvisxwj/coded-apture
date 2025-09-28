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
from Norm import *
import scipy.io as sio
import h5py
from PIL import Image

def minmax_norm(channel):
    c_min = channel.min()
    c_max = channel.max()
    if abs(c_max - c_min) < 1e-12:
        return np.zeros_like(channel)  # 避免除以0
    return (channel - c_min) / (c_max - c_min)


# 自定义测试数据集，现在针对新的两个特定文件进行处理，不做数据增强
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_path, label_path, is_Train=False):
        super(TestDataset, self).__init__()
        self.hsi_path = hsi_path
        self.label_path = label_path
        self.is_Train = is_Train  # 新增参数用于区分训练集和测试集

        # 加载HSI数据
        with h5py.File(self.hsi_path, 'r') as f:
            # 假设 'extracted_data' 是你要读取的数据集名称
            self.hsi = f['extracted_data'][:] / 2600.0  # 数据归一化
            self.hsi = self.hsi.transpose(2, 1, 0)
        print('hsi.shape', self.hsi.shape)

        # 加载label数据
        with h5py.File(self.label_path, 'r') as f:
            # 假设 'spectral_indices' 是你要读取的数据集名称
            self.label = f['spectral_indices'][:]
            self.label = self.label.transpose(2, 1, 0)
        print('label.shape', self.label.shape)

        _, width, _ = self.hsi.shape  # 获取数据宽度维度大小
        cut_index = int(width * 0.8)  # 计算截取的索引位置，80%处

        # 根据是训练集还是测试集进行不同的截取操作
        if self.is_Train:
            self.hsi = self.hsi[:, :cut_index, :]
            self.label = self.label[:, :cut_index, :]
        else:
            self.hsi = self.hsi[:, cut_index:, :]
            self.label = self.label[:, cut_index:, :]

        print('hsi.shape',self.hsi.shape)
        print('label.shape',self.label.shape)
        # 加载mask，这里假设原mask路径及读取方式不变，如果实际有变需调整
        mat_data = sio.loadmat(r"/root/autodl-tmp/mask_sim.mat")
        mask = mat_data['mask']  # 原始 (256, 256)
        print(f"Original Shape of mask: {mask.shape}")

        # 将 mask 的长宽各复制为原来的5倍（原逻辑，若实际不需要可去掉）
        self.mask = np.repeat(np.repeat(mask, 5, axis=0), 5, axis=1)
        print(f"Resized Shape of mask: {self.mask.shape}")

        # 沿第三维度复制84次（原逻辑，若实际不需要可去掉）
        mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

        # self.mask_3d = mask_3d[:1072, :1004, :]
        self.mask_3d = mask_3d
        print(f"Shape of mask_3d after cropping: {self.mask_3d.shape}")

        start_H = 700  # TODO，这里如果裁剪逻辑需要调整按实际情况改
        start_W = 200
        # 新增的500x500裁剪 ，原逻辑保留，可按需调整
        self.hsi = self.hsi[start_H:start_H + 256, start_W:start_W + 256, :]
        self.label = self.label[start_H:start_H + 256, start_W:start_W + 256, :]
        self.mask_3d = self.mask_3d[start_H:start_H + 256, start_W:start_W + 256, :]
        
        # 保存为RGB图像
        input_sample = self.hsi
        print('input_sample', input_sample.shape)
        # Chikusei保存波段
        r = input_sample[:, :, 45]
        g = input_sample[:, :, 24]
        b = input_sample[:, :, 3]
        '''
        # XiongAn保存波段
        r = input_sample[:, :, 46]
        g = input_sample[:, :, 23]
        b = input_sample[:, :, 6]
        '''
        # 分别归一化
        r_norm = minmax_norm(r)
        g_norm = minmax_norm(g)
        b_norm = minmax_norm(b)

        rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
        rgb_image = (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_image, mode='RGB')
        # TODO
        img.save(f"/root/Code/Sample/XiongAn/RGB.png")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        hsi = self.hsi
        target = self.label
        mask_3d = self.mask_3d
        shape = hsi.shape

        # 生成测量帧，原逻辑保留，可按需调整
        temp = mask_3d * hsi
        temp_shift = np.zeros((shape[0], shape[1] + (84 - 1) * 2, 84))
        temp_shift[:, 0:shape[1], :] = temp
        mask_3d_shift = np.zeros((shape[0], shape[1] + (84 - 1) * 2, 84))
        mask_3d_shift[:, 0:shape[1], :] = mask_3d

        for t in range(84):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
            mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
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
# TODO
model = torch.load(
    "/root/Code/parameter/XiongAn/UNet/UNetW4C_lr1-3_bs1_Epoch100_2560_hybridLoss.pth", map_location="cuda:0")
model.to(device)

loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
loss_SSIM = SSIM()

hsi_filename = r"/root/autodl-tmp/Public_Data/XiongAn_jg5.mat"
label_filename = r"/root/autodl-tmp/Public_Data/new_XiongAn_indices.mat"

# 创建测试数据集和数据加载器，传入is_Train=False表示是测试集情况，按要求截取右边20%数据
test_dataset = TestDataset(hsi_filename, label_filename, is_Train=False)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print('outputs.shape', outputs.shape)
        
        # 将label也保存下来，以不同的文件名
        labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
        # TODO
        save_path_label = "/root/Code/Sample/XiongAn/UNet/label.mat"
        sio.savemat(save_path_label, {"label": labels_np})
        print(f"Saved label to {save_path_label}")
        
        # TODO Normalize outputs and labels
        labels = XA_max_min_norm(labels)

        # Calculate losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
        L1_loss = loss_L1(outputs, labels)
        SSIM_loss = loss_SSIM(outputs, labels)
        loss = Charbonnier_loss + L1_loss + SSIM_loss
         
        print (f"Charbonnier Loss: {Charbonnier_loss:.4f}, L1 Loss: {L1_loss:.4f}, SSIM Loss: {SSIM_loss:.4f}, Total Loss: {loss:.4f}")

        # TODO
        outputs = reverse_XA_max_min_norm(outputs)
        # 将输出保存为mat文件
        outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
        # TODO
        save_path_output = "/root/Code/Sample/XiongAn/UNet/result.mat"
        sio.savemat(save_path_output, {"output": outputs_np})
        print(f"Saved result to {save_path_output}")
'''
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        print('outputs.shape', outputs.shape)

        # 将输出保存为mat文件
        outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
        save_path_output = "/data4/zhuo-file/fyc_file/ATTU_Norm/Sample/public/XiongAn/ATTU/result.mat"
        sio.savemat(save_path_output, {"output": outputs_np})
        print(f"Saved result to {save_path_output}")

        # Normalize outputs and labels
        labels = XA_max_min_norm(labels)
        
        # 将label也保存下来，以不同的文件名
        labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
        save_path_label = "/data4/zhuo-file/fyc_file/ATTU_Norm/Sample/public/XiongAn/ATTU/label.mat"
        sio.savemat(save_path_label, {"label": labels_np})
        print(f"Saved label to {save_path_label}")
        
        # Calculate losses
        Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
        L1_loss = loss_L1(outputs, labels)
        SSIM_loss = loss_SSIM(outputs, labels)
        loss = Charbonnier_loss + L1_loss + SSIM_loss

        print(f"Charbonnier Loss: {Charbonnier_loss:.4f}, L1 Loss: {L1_loss:.4f}, SSIM Loss: {SSIM_loss:.4f}, Total Loss: {loss:.4f}")
'''