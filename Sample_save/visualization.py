import os
import scipy.io as sio
import numpy as np
from PIL import Image

file_path = '/data4/zhuo-file/fyc_file/Arti11/Sample/ours_best150_0_0/'
model_name = 'UNet_KAN'
# 定义文件路径和保存目录
file_dict = {
    "label": {
        "path": f"{file_path}{model_name}/150_best_label.mat",
        "save_dir": f"{file_path}{model_name}/Origin_Figure/"
    },
    "result": {
        "path": f"{file_path}{model_name}/150_best_result.mat",
        "save_dir": f"/{file_path}{model_name}/Predicted_Figure/"
    }
}

for data_type, info in file_dict.items():
    mat_file = info["path"]
    save_dir = info["save_dir"]
    # 如果目录不存在，则创建
    os.makedirs(save_dir, exist_ok=True)
    # 读取mat文件
    data = sio.loadmat(mat_file)
    if data_type == "label":
        key = "label"
    else:
        key = "output"
    output = data[key]

    # 输出维度检查
    print(f"{data_type} output shape:", output.shape)

    C, H, W = output.shape

    # 遍历每个通道，将其可视化为单通道灰度图像
    for i in range(C):
        # 取出第i个通道的数据
        channel_data = output[i, :, :]

        # 打印通道数据的最小值和最大值
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        print(f"{data_type} Channel {i + 1} min: {channel_min}, max: {channel_max}")

        # 检查是否所有值相同
        if channel_max == channel_min:
            print(f"{data_type} Channel {i + 1} has all identical values: {channel_min}")
            channel_norm = np.zeros_like(channel_data)
        else:
            # 将数据归一化到[0, 1]
            channel_norm = (channel_data - channel_min) / (channel_max - channel_min)

        # 将[0, 1]的数据扩展到[0, 255]并转换为uint8
        channel_img = (channel_norm * 255).astype(np.uint8)

        # 打印归一化后的通道数据的最小值和最大值
        print(f"{data_type} Normalized Channel {i + 1} min: {channel_img.min()}, max: {channel_img.max()}")

        # 使用PIL保存为单通道灰度图像
        img = Image.fromarray(channel_img, mode='L')
        save_path = os.path.join(save_dir, f"channel_{i + 1}.png")
        img.save(save_path)

print("All channel images for label and result have been saved!")