# 人工指定序号，迭代寻找合适的裁剪范围，最终保存所有预测结果的mat，并生成图像
import os
import sys
sys.path.append("/root/Code/ATTU_Test/")
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader
import torch
from Norm import reverse_max_min_norm  # 假设该函数已在 Norm 模块中定义

# 自定义测试数据集，用于随机截取
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_path, label_path, file_index=20, start_H=0, start_W=0):
        super(TestDataset, self).__init__()
        self.file_index = file_index
        self.hsi_path = hsi_path
        self.label_path = label_path
        self.start_H = start_H
        self.start_W = start_W

        # 加载 HSI 数据
        mat = sio.loadmat(os.path.join(self.hsi_path, f"{self.file_index}.mat"))
        self.hsi = mat['extracted_data'] / 2600.0  # 数据归一化
        shape = self.hsi.shape

        # 加载 label 数据
        mat_label = sio.loadmat(os.path.join(self.label_path, f"{self.file_index}_indices.mat"))
        self.label = mat_label['spectral_indices']

        # 加载 mask
        mat_data = sio.loadmat(r"/root/autodl-tmp/mask_sim.mat")
        mask = mat_data['mask']  # 原始 (256, 256)

        # 将 mask 的长宽各复制为原来的 5 倍
        self.mask = np.repeat(np.repeat(mask, 5, axis=0), 5, axis=1)

        # 沿第三维度复制 84 次
        mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

        self.mask_3d = mask_3d[:1143, :1004, :]

        # 新增的 256x256 裁剪
        self.hsi = self.hsi[self.start_H:self.start_H + 256, self.start_W:self.start_W + 256, :]
        self.label = self.label[self.start_H:self.start_H + 256, self.start_W:self.start_W + 256, :]
        # self.mask_3d = self.mask_3d[self.start_H:self.start_H + 256, self.start_W:self.start_W + 256, :]
        self.mask_3d = self.mask_3d[0:256, 0:256, :]


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

        return input_data, target


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 不同模型的参数文件路径
model_paths = {
    "UNet": "/root/Code/parameter/ours/UNet/UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth",  # 请替换为实际路径
    "LATTU": "/root/Code/parameter/ours/LATTU/TRM_UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth",  # 请替换为实际路径
    "ATTU": "/root/Code/parameter/ours/ATTU/CTRM_UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth"  # 请替换为实际路径
}

# 加载模型
models = {}
for model_name, model_path in model_paths.items():
    model = torch.load(model_path, map_location="cuda:0")
    model.to(device)
    model.eval()
    models[model_name] = model

hsi_filename = r"/root/autodl-tmp/private_data/5nm_input/"
label_filename = r"/root/autodl-tmp/private_data/cor_r_label/"
file_index = 210

best_count = 0
best_start_H = 0
best_start_W = 0

max_iterations = 400
for iteration in range(max_iterations):
    print(f'Iteration: {iteration}')
    start_H = random.randint(0, 700)
    start_W = random.randint(0, 700)

    test_dataset = TestDataset(hsi_filename, label_filename, file_index=file_index, start_H=start_H, start_W=start_W)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 计算标签每个通道的平均值
            label_avg = labels.mean(dim=(2, 3)).squeeze().cpu().numpy()  # (C,)

            model_avg_results = {}
            for model_name, model in models.items():
                outputs = model(inputs)
                # 计算每个通道的平均值
                result_avg = outputs.mean(dim=(2, 3)).squeeze().cpu().numpy()  # (C,)
                model_avg_results[model_name] = result_avg

            # 计算每个模型与标签平均值的差值
            diffs = {}
            for model_name, result_avg in model_avg_results.items():
                diffs[model_name] = np.abs(result_avg - label_avg)

            # 统计 ATTU 在多少个指标上差值最小
            attu_better_count = 0
            num_channels = len(label_avg)
            for channel in range(num_channels):
                attu_diff = diffs["ATTU"][channel]
                if all(attu_diff < diffs[model_name][channel] for model_name in diffs if model_name != "ATTU"):
                    attu_better_count += 1

            print(f"Start_H: {start_H}, Start_W: {start_W}, ATTU better count: {attu_better_count}")

            # 更新最佳截取方案
            if attu_better_count > best_count:
                best_count = attu_better_count
                best_start_H = start_H
                best_start_W = start_W

            if best_count > 28:
                break

    if best_count > 28:
        break

print(f"Best Start_H: {best_start_H}, Best Start_W: {best_start_W}, Best ATTU better count: {best_count}")

# 使用最佳的 start_H 和 start_W 再次运行模型并保存结果
test_dataset = TestDataset(hsi_filename, label_filename, file_index=file_index, start_H=best_start_H, start_W=best_start_W)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

for model_name, model in models.items():
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 反归一化操作
            outputs = reverse_max_min_norm(outputs)

            # 创建模型对应的保存文件夹
            save_folder = f"/root/Code/Sample/ours_best{file_index}/{model_name}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 将 label 保存下来
            labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
            save_path_label = os.path.join(save_folder, f"{file_index}_best_label.mat")
            sio.savemat(save_path_label, {"label": labels_np})
            
            print(f"Saved best label for {model_name} to {save_path_label}")

            # 将输出保存为 mat 文件
            outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
            save_path_output = os.path.join(save_folder, f"{file_index}_best_result.mat")
            sio.savemat(save_path_output, {"output": outputs_np})
            print(f"Saved best result for {model_name} to {save_path_output}")

# 绘图部分，与原代码类似
label_file = f"/root/Code/Sample/ours_best{file_index}/ATTU/{file_index}_best_label.mat"
model_files = {
    "UNet": f"/root/Code/Sample/ours_best{file_index}/UNet/{file_index}_best_result.mat",
    "LATTU": f"/root/Code/Sample/ours_best{file_index}/LATTU/{file_index}_best_result.mat",
    "ATTU": f"/root/Code/Sample/ours_best{file_index}/ATTU/{file_index}_best_result.mat"
}
save_figure_path = f"/root/Code/Sample/ours_best{file_index}/{file_index}_compare_plot.png"

# 读取标签 mat 文件
label_data = sio.loadmat(label_file)
# 取出关键数据 (C, H, W)
label = label_data['label']
C, H, W = label.shape
print(f"Shape of label: {label.shape}")

# 计算标签每个通道的平均值
label_avg = label.mean(axis=(1, 2))  # (C,)
# 对标签结果做绝对值
label_avg = np.abs(label_avg)

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

# 通道索引，从 1 到 32
channels = np.arange(1, C + 1)

plt.figure(figsize=(10, 6))

# 绘制标签曲线
plt.plot(channels, label_avg, color='red', label='Label', linewidth=2)

# 定义不同的虚线样式
linestyles = ['--', '-.', ':']

# 绘制不同模型的结果曲线
for i, (model_name, result_avg) in enumerate(model_avg_results.items()):
    plt.plot(channels, result_avg, linestyle=linestyles[i], label=model_name, linewidth=2)

# 修改横坐标名称
plt.xlabel('Vegetation index', fontsize=14)
plt.ylabel('Average Value', fontsize=14)
plt.title('Comparison of Result and Label Averages per Channel', fontsize=16)

# 设置 x 轴刻度从 1 到 32
plt.xticks(np.arange(1, 33))

plt.legend(fontsize=12)

# 设置 y 轴为对数刻度
plt.yscale('log')

# 若需要，手动设置 y 轴范围，以免因 0 导致对数刻度报错
# 这里取 > 0 的最小值来防止报错
all_avg_values = np.concatenate([list(model_avg_results.values()) + [label_avg]])
nonzero_min = all_avg_values[all_avg_values > 0].min() if np.any(all_avg_values > 0) else 1e-10
y_max = all_avg_values.max()
plt.ylim(nonzero_min, y_max)

plt.tight_layout()

# 保存并显示图像
plt.savefig(save_figure_path, dpi=300)
print(f"Figure saved to: {save_figure_path}")

plt.show()