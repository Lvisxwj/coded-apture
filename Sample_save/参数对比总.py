import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# =============== 1. 文件路径与读取 ===============
# 标签文件路径
label_file  = "/root/Code/Sample/ours220/ATTU/label.mat"
# 不同模型的结果文件路径
model_files = {
    "UNet": "/root/Code/Sample/ours220/UNet/result.mat",
    "LATTU": "/root/Code/Sample/ours220/LATTU/result.mat",
    "ATTU": "/root/Code/Sample/ours220/ATTU/result.mat"
}
save_figure_path = "/root/Code/Sample/ours220/compare_plot.png"  # 最终要保存的图像文件名

# 读取标签 mat 文件
label_data = sio.loadmat(label_file)
# 取出关键数据 (C, H, W)
label  = label_data['label']
C, H, W = label.shape
print(f"Shape of label: {label.shape}")

# 计算标签每个通道的平均值
label_avg  = label.mean(axis=(1, 2))   # (C,)
# 对标签结果做绝对值
label_avg  = np.abs(label_avg)

# 存储不同模型的平均结果
model_avg_results = {}

# 遍历不同模型的结果文件
for model_name, result_file in model_files.items():
    # 读取结果 mat 文件
    result_data = sio.loadmat(result_file)
    # 取出关键数据 (C, H, W)
    result = result_data['output']
    print(f"Shape of {model_name} result: {result.shape}")

    # 计算每个通道的平均值
    result_avg = result.mean(axis=(1, 2))  # (C,)
    # 对结果做绝对值
    result_avg = np.abs(result_avg)

    # 存储该模型的平均结果
    model_avg_results[model_name] = result_avg

# =============== 4. 绘图部分 ===============
# 通道索引
channels = np.arange(1, C+1)

plt.figure(figsize=(10, 6))

# 绘制标签曲线
plt.plot(channels, label_avg,  color='red',  label='Label',  linewidth=2)

# 定义不同的虚线样式
linestyles = ['--', '-.', ':']

# 绘制不同模型的结果曲线
for i, (model_name, result_avg) in enumerate(model_avg_results.items()):
    plt.plot(channels, result_avg, linestyle=linestyles[i], label=model_name, linewidth=2)

plt.xlabel('Channel', fontsize=14)
plt.ylabel('Average Value', fontsize=14)
plt.title('Comparison of Result and Label Averages per Channel', fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置 y 轴为对数刻度
plt.yscale('log')

# 若需要，手动设置 y 轴范围，以免因 0 导致对数刻度报错
# 这里取 > 0 的最小值来防止报错
all_avg_values = np.concatenate([list(model_avg_results.values()) + [label_avg]])
nonzero_min = all_avg_values[all_avg_values > 0].min() if np.any(all_avg_values > 0) else 1e-10
y_max = all_avg_values.max()
plt.ylim(nonzero_min, y_max)

plt.tight_layout()

# =============== 5. 保存并显示图像 ===============
plt.savefig(save_figure_path, dpi=300)
print(f"Figure saved to: {save_figure_path}")

plt.show()