import numpy as np
from PIL import Image, ImageDraw
import h5py
import os

# 公共数据集文件路径
data_file_path = r"/root/autodl-tmp/Public_Data/Chikusei_jg5.mat"

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

# 转换为带透明度通道的图像
img = img.convert("RGBA")

# 定义三个框的左上角坐标（相对于右侧20%区域）W,H
boxes = [(20, 665), (75, 1908), (33, 1209)]
box_size = 256
transparency = 64  # 透明度，0 完全透明，255 完全不透明
border_width = 4  # 边框粗细

# 计算右侧20%区域的起始横坐标
right_start = cut_index

# 调整框的坐标为相对于整张图像的坐标
adjusted_boxes = [(x + right_start, y) for x, y in boxes]

# 创建绘图对象
draw = ImageDraw.Draw(img)

# 绘制三个框
for x, y in adjusted_boxes:
    box_coords = [(x, y), (x + box_size, y + box_size)]
    # 绘制白色半透明矩形边框
    draw.rectangle(box_coords, fill=None, outline=(255, 255, 255, transparency), width=border_width)

# 保存图像
save_path = f"/root/Code/Sample/Chikusei_best/Whole_RGB_with_boxes_public.png"
img.save(save_path)
print(f"Image saved to {save_path}")