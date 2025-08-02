import scipy.io as sio
import numpy as np
from PIL import Image, ImageDraw

index = 220
hsi_filename = r"/root/autodl-tmp/private_data/5nm_input/"

def minmax_norm(channel):
    c_min = channel.min()
    c_max = channel.max()
    if abs(c_max - c_min) < 1e-12:
        return np.zeros_like(channel)  # 避免除以0
    return (channel - c_min) / (c_max - c_min)

# 加载 HSI 数据
mat = sio.loadmat(f"{hsi_filename}{index}.mat")
hsi = mat['extracted_data'] / 2600.0  # 数据归一化

# 保存为 RGB 图像
input_sample = hsi
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

# 转换为带透明度通道的图像
img = img.convert("RGBA")

# 定义三个框的左上角坐标W,H
boxes = [(661, 542), (590, 26), (105, 331)]
box_size = 256
transparency = 64  # 透明度，0 完全透明，255 完全不透明
border_width = 4  # 边框粗细

# 创建绘图对象
draw = ImageDraw.Draw(img)

# 绘制三个框
for x, y in boxes:
    box_coords = [(x, y), (x + box_size, y + box_size)]
    # 绘制白色半透明矩形边框
    draw.rectangle(box_coords, fill=None, outline=(255, 255, 255, transparency), width=border_width)

# 保存图像
img.save(f"/root/Code/Sample/ours220/Whole_RGB_with_boxes.png")