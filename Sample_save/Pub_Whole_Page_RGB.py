import numpy as np
from PIL import Image
import h5py
import os

# 公共数据集文件路径
data_file_path = r"/root/autodl-tmp/Public_Data/XiongAn_jg5.mat"

def minmax_norm(channel):
    c_min = channel.min()
    c_max = channel.max()
    if abs(c_max - c_min) < 1e-12:
        return np.zeros_like(channel)  # 避免除以0
    return (channel - c_min) / (c_max - c_min)

# 使用 h5py 加载 HSI 数据
with h5py.File(data_file_path, 'r') as f:
    hsi = f['extracted_data'][:] / 2600.0  # 数据归一化
    hsi = hsi.transpose(2, 1, 0)
print(hsi.shape)
height, width, _ = hsi.shape  # 获取数据高度和宽度维度大小
cut_index = int(width * 0.8)  # 计算截取的索引位置，80%处

# 保存为 RGB 图像
input_sample = hsi
print('input_sample', input_sample.shape)
'''
# Chikusei保存波段
r = input_sample[:, :, 45]
g = input_sample[:, :, 24]
b = input_sample[:, :, 3]
'''
# XiongAn保存波段
r = input_sample[:, :, 46]
g = input_sample[:, :, 23]
b = input_sample[:, :, 6]

# 分别归一化
r_norm = minmax_norm(r)
g_norm = minmax_norm(g)
b_norm = minmax_norm(b)
rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
rgb_image = (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
img = Image.fromarray(rgb_image, mode='RGB')

# 保存图像
save_path = f"/root/Code/Sample/XiongAn_best/Whole_RGB.png"
img.save(save_path)
print(f"Image saved to {save_path}")