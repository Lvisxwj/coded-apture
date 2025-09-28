from torch.autograd import Variable
import sys
sys.path.append("/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/")
import torch
from torch.utils.data import DataLoader
from Model import UNet
from Model import *
from lz_dataset import dataset
import datetime
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import numpy as np
from loss import *
from torch.nn.utils import clip_grad_norm_
from Norm import *

hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/cor_r_label/"
# 加载数据和网络
Dataset = dataset(hsi_filename, label_filename)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(
    "/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/parameter/ours/UNet_KAN_KAN/UNet_KAN_KAN_lr1-3_bs1_Epoch200_2560_hybridLoss.pth",map_location="cuda:0")  # 要改为要测试的模型地址
model.to(device)

# 超参数
batch_size = 1
Max_epoch = 5
test_set = 2560  # 和dataset.py中的test_set一致

# 构建损失函数和优化器
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss()
# loss_PSNR = psnr()
loss_SSIM = SSIM()
loss_PSNR = PSNR()


if __name__ == "__main__":
    Whole_Loss = []
    log_file = r"./testing_log.txt"
    with open(log_file, "w") as f:
        for epoch in range(1, Max_epoch + 1):

            model.eval()
            test_loader = DataLoader(Dataset, num_workers=8, batch_size=batch_size, shuffle=False)

            # 每个epoch的loss
            epoch_loss = 0.0
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0
            epoch_PSNR_value = 0.0

            start_time = time.time()

            for i, (inputs, labels) in enumerate(test_loader):  # i代表第i次迭代

                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    k, outputs = model(inputs)  # [bs, 4, 64, 64]
                # print(labels.shape, outputs.shape)

                # outputs = max_min_norm(outputs)
                labels = max_min_norm(labels)

                # 计算损失
                """loss = loss_CharbonnierLoss(outputs, labels)"""

                Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
                L1_loss = loss_L1(outputs, labels)
                SSIM_loss = loss_SSIM(outputs, labels)
                loss = Charbonnier_loss + L1_loss + SSIM_loss
                PSNR_value = loss_PSNR(outputs, labels)

                epoch_loss += loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()
                epoch_PSNR_value += PSNR_value.item()

                # 反向传播
                # optimizer.zero_grad()
                # loss.backward()

                clip_grad_norm_(model.parameters(), max_norm=0.2)

                # optimizer.step()
                # scheduler.step()

                # 损失可视化
                if i % (100) == 0:
                    log_str = '%4d %4d / %4d whole_loss = %.10f time = %s\n' % (
                        epoch, i, test_set // batch_size, epoch_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d Charbonnierloss = %.10f time = %s\n' % (
                        epoch, i, test_set // batch_size, epoch_Charbonnier_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d L1_loss = %.10f time = %s\n' % (
                        epoch, i, test_set // batch_size, epoch_L1_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d SSIM_loss = %.10f time = %s\n' % (
                        epoch, i, test_set // batch_size, epoch_SSIM_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d PSNR = %.10f time = %s\n' % (
                        epoch, i, test_set // batch_size,
                        epoch_PSNR_value / (i + 1),
                        datetime.datetime.now()
                    )

                    log_str += '\n'  # 添加一个空行
                    f.write(log_str)
                    f.flush()
                    print(log_str)

            elapsed_time = time.time() - start_time
            epoch_log_str = 'epoch = %4d , loss = %.10f , time = %4.2f s\n' % (
                epoch, epoch_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , Charbonnierloss = %.10f , time = %4.2f s\n' % (
                epoch, epoch_Charbonnier_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , L1_loss = %.10f , time = %4.2f s\n' % (
                epoch, epoch_L1_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , SSIM_loss = %.10f , time = %4.2f s\n' % (
                epoch, epoch_SSIM_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , PSNR = %.10f , time = %4.2f s\n' % (
                epoch, epoch_PSNR_value / (test_set / batch_size), elapsed_time)
            
            
            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str)

            # loss
            Whole_Loss.append(epoch_L1_loss / (test_set / batch_size))

            f.write('save success\n')
            f.flush()
            print('save success\n')

    # 保存loss
    # np.savetxt('./Loss.csv', Whole_Loss, delimiter=',')

    # 画出loss的散点图
    epoch = [i for i in range(1, Max_epoch + 1)]
    plt.scatter(epoch, Whole_Loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # 保存loss的散点图
    # plt.savefig('./Loss.png')
