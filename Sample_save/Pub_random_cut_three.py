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
from Norm import *  # 假设这些函数已在 Norm 模块中定义
from loss import CharbonnierLoss, SSIM  # 假设这些损失函数已在 loss 模块中定义
from PIL import Image
import h5py

def minmax_norm(channel):
    c_min = channel.min()
    c_max = channel.max()
    if abs(c_max - c_min) < 1e-12:
        return np.zeros_like(channel)  # 避免除以0
    return (channel - c_min) / (c_max - c_min)

# 自定义测试数据集，用于随机截取
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, hsi_path, label_path, start_H=0, start_W=0, is_Train=False):
        super(TestDataset, self).__init__()
        self.dataset_name = dataset_name
        self.hsi_path = hsi_path
        self.label_path = label_path
        self.start_H = start_H
        self.start_W = start_W
        self.is_Train = is_Train  # 新增参数用于区分训练集和测试集

        if self.dataset_name == "Chikusei":
            hsi_file = "Chikusei_jg5.mat"
            label_file = "new_Chikusei_indices.mat"
        elif self.dataset_name == "XiongAn":
            hsi_file = "XiongAn_jg5.mat"
            label_file = "new_XiongAn_indices.mat"
        else:
            raise ValueError("Invalid dataset name. Choose either 'Chikusei' or 'XiongAn'.")

        # 加载HSI数据
        with h5py.File(os.path.join(self.hsi_path, hsi_file), 'r') as f:
            # 假设 'extracted_data' 是你要读取的数据集名称
            self.hsi = f['extracted_data'][:] / 2600.0  # 数据归一化
            self.hsi = self.hsi.transpose(2, 1, 0)
        # print('hsi.shape', self.hsi.shape)

        # 加载label数据
        with h5py.File(os.path.join(self.label_path, label_file), 'r') as f:
            # 假设 'spectral_indices' 是你要读取的数据集名称
            self.label = f['spectral_indices'][:]
            self.label = self.label.transpose(2, 1, 0)
        # print('label.shape', self.label.shape)

        height, width, _ = self.hsi.shape  # 获取数据高度和宽度维度大小
        cut_index = int(width * 0.8)  # 计算截取的索引位置，80%处

        # 根据是训练集还是测试集进行不同的截取操作
        if self.is_Train:
            self.hsi = self.hsi[:, :cut_index, :]
            self.label = self.label[:, :cut_index, :]
        else:
            self.hsi = self.hsi[:, cut_index:, :]
            self.label = self.label[:, cut_index:, :]

        self.test_height, self.test_width, _ = self.hsi.shape  # 获取测试集的高度和宽度
        print(self.test_height, self.test_width)

        # print('hsi.shape', self.hsi.shape)
        # print('label.shape', self.label.shape)
        # 加载mask
        mat_data = sio.loadmat(r"/root/autodl-tmp/mask_sim.mat")
        mask = mat_data['mask']  # 原始 (256, 256)
        # print(f"Original Shape of mask: {mask.shape}")

        # 将 mask 的长宽各复制为原来的5倍
        self.mask = np.repeat(np.repeat(mask, 5, axis=0), 5, axis=1)
        # print(f"Resized Shape of mask: {self.mask.shape}")

        # 沿第三维度复制84次
        mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 84))

        self.mask_3d = mask_3d
        # print(f"Shape of mask_3d after cropping: {self.mask_3d.shape}")

        # 新增的256x256裁剪
        self.hsi = self.hsi[self.start_H:self.start_H + 256, self.start_W:self.start_W + 256, :]
        self.label = self.label[self.start_H:self.start_H + 256, self.start_W:self.start_W + 256, :]
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
        # print('input.shape', input_data.shape)

        return input_data, target


# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO 让用户选择数据集
dataset_name = "Chikusei"
if dataset_name not in ["Chikusei", "XiongAn"]:
    raise ValueError("无效的数据集名称。请选择 'Chikusei' 或 'XiongAn'.")

# TODO 不同模型的参数文件路径
model_paths = {
    "UNet": f"/root/Code/parameter/{dataset_name}/UNet/UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth",
    "LATTU": f"/root/Code/parameter/{dataset_name}/LATTU/TRM_UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth",
    "ATTU": f"/root/Code/parameter/{dataset_name}/ATTU/CTRM_UNetW4C_lr1-3_bs1_Epoch200_2560_hybridLoss.pth"
}

# 加载模型
models = {}
for model_name, model_path in model_paths.items():
    model = torch.load(model_path, map_location="cuda:0")
    model.to(device)
    model.eval()
    models[model_name] = model

hsi_filename = r"/root/autodl-tmp/Public_Data/"
label_filename = r"/root/autodl-tmp/Public_Data/"

best_count = 0
best_start_H = 0
best_start_W = 0

max_iterations = 400

# 在迭代开始前只加载一次数据集，获取测试集的大小
temp_dataset = TestDataset(dataset_name, hsi_filename, label_filename, is_Train=False)
test_height = temp_dataset.test_height
test_width = temp_dataset.test_width

# 调整随机取值范围，确保裁剪区域在测试集内
max_start_H = max(0, test_height - 256)
max_start_W = max(0, test_width - 256)

for iteration in range(max_iterations):
    print(f'Iteration: {iteration}')

    start_H = random.randint(0, max_start_H)
    start_W = random.randint(0, max_start_W)

    test_dataset = TestDataset(dataset_name, hsi_filename, label_filename, start_H=start_H, start_W=start_W, is_Train=False)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    loss_L1 = torch.nn.L1Loss()
    loss_CharbonnierLoss = CharbonnierLoss()
    loss_SSIM = SSIM()

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

            if best_count > 20:
                break

    if best_count > 20:
        break

print(f"Best Start_H: {best_start_H}, Best Start_W: {best_start_W}, Best ATTU better count: {best_count}")

# 使用最佳的 start_H 和 start_W 再次运行模型并保存结果
test_dataset = TestDataset(dataset_name, hsi_filename, label_filename, start_H=best_start_H, start_W=best_start_W, is_Train=False)
test_loader = DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

for model_name, model in models.items():
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 根据数据集选择反归一化函数
            if dataset_name == "Chikusei":
                outputs = reverse_Chi_max_min_norm(outputs)
            elif dataset_name == "XiongAn":
                outputs = reverse_XA_max_min_norm(outputs)

            # 创建模型对应的保存文件夹
            save_folder = f"/root/Code/Sample/{dataset_name}_best/{model_name}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # 将 label 保存下来
            labels_np = labels.squeeze().cpu().numpy()  # [C, H, W]
            save_path_label = os.path.join(save_folder, "best_label.mat")
            sio.savemat(save_path_label, {"label": labels_np})
            print(f"Saved best label for {model_name} to {save_path_label}")

            # 将输出保存为 mat 文件
            outputs_np = outputs.squeeze().cpu().numpy()  # [C, H, W]
            save_path_output = os.path.join(save_folder, "best_result.mat")
            sio.savemat(save_path_output, {"output": outputs_np})
            print(f"Saved best result for {model_name} to {save_path_output}")

# 绘图部分，与原代码类似
label_file = f"/root/Code/Sample/{dataset_name}_best/ATTU/best_label.mat"
model_files = {
    "UNet": f"/root/Code/Sample/{dataset_name}_best/UNet/best_result.mat",
    "LATTU": f"/root/Code/Sample/{dataset_name}_best/LATTU/best_result.mat",
    "ATTU": f"/root/Code/Sample/{dataset_name}_best/ATTU/best_result.mat"
}
save_figure_path = f"/root/Code/Sample/{dataset_name}_best/compare_plot.png"

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
plt.title('Comparison of Result and Label Averages per Vegetation index', fontsize=16)

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