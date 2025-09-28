import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 文件路径
result_file = "/data4/zhuo-file/fyc_file/ATTU_Norm/Sample/ours/ATTU/result.mat"
label_file = "/data4/zhuo-file/fyc_file/ATTU_Norm/Sample/ours/ATTU/label.mat"
# save_figure_path = "/data4/zhuo-file/fyc_file/ATTU_KAN_Norm_copy/Sample/210/XiongAn_compare_plot.png"

# 读取mat文件
result_data = sio.loadmat(result_file)
label_data = sio.loadmat(label_file)

result = result_data['output']  # (C, H, W)
label = label_data['label']     # (C, H, W)

C, H, W = result.shape

# 计算平均值
result_avg = result.mean(axis=(1, 2))
label_avg = label.mean(axis=(1, 2))

# 输出平均值
print("Result per-channel averages:", result_avg)
print("Label per-channel averages:", label_avg)

