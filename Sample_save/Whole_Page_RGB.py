import scipy.io as sio
import numpy as np
from PIL import Image

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
# 降低蓝色通道权重，这里乘以0.7，可根据实际情况调整
b_norm = b_norm * 0.7
rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
rgb_image = (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
img = Image.fromarray(rgb_image, mode='RGB')
img.save(f"/root/Code/Sample/ours220/Whole_RGB.png")