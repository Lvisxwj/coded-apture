from torch.autograd import Variable
import sys
sys.path.append("/data4/zhuo-file/fyc_newfile/Arti11/ATTU_Norm/")
import torch
from torch.utils.data import DataLoader
from Model import *
from dataset import dataset
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
#加载数据和网络
Dataset = dataset(hsi_filename, label_filename)

file_path = "/data4/zhuo-file/fyc_newfile/Arti11/ATTU_Norm/"
#Todo
model = UNet()
Model_name = "UNet"
log_file = rf"{file_path}parameter/ours/training_log_{Model_name}_lr4-4_bs8_Epoch500_2560_hybridLoss.txt"

#GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

#超参数
# learning_rate = 0.001
learning_rate = 0.0004
# milestones = [60, 120, 180, 240, 300, 360]
milestones = [100, 200, 300, 400]
gamma = 0.5
# batch_size = 1
batch_size = 8
# Max_epoch = 300
Max_epoch = 500
train_set = 2560 #和dataset.py中的train_set一致

# 构建损失函数和优化器
loss_CharbonnierLoss = CharbonnierLoss() 

# optimizing
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if __name__ == "__main__": 
    print(learning_rate, Max_epoch)   
    Whole_Loss = []
    with open(log_file, "w") as f:
        for epoch in range(1, Max_epoch + 1):

            #模型训练
            model.train()
            train_loader = DataLoader(Dataset, num_workers=8, batch_size=batch_size, shuffle=True)

            #每个epoch的loss
            epoch_loss = 0.0 
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0

            start_time = time.time()

            for i, (hsi, inputs, labels) in enumerate(train_loader): #i代表第i次迭代
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # 归一化
                # outputs = max_min_norm(outputs)
                labels = max_min_norm(labels)

                #计算损失
                """loss = loss_CharbonnierLoss(outputs, labels)"""
                
                loss = loss_CharbonnierLoss(outputs, labels) 
                epoch_loss += loss.item()

                #反向传播
                optimizer.zero_grad()
                loss.backward()

                
                clip_grad_norm_(model.parameters(), max_norm=0.2)

                optimizer.step()
                scheduler.step()

                #损失可视化
                if i % (100) == 0:
                    log_str = '%4d %4d / %4d Charbonnierloss = %.10f time = %s\n' % (epoch, i, train_set // batch_size, epoch_loss / (i + 1),
                        datetime.datetime.now())
                    
                    # log_str += '\n'  # 添加一个空行
                    f.write(log_str)
                    f.flush()
                    print(log_str)                    

            elapsed_time = time.time() - start_time
            epoch_log_str = 'epoch = %4d , Charbonnierloss = %.10f , time = %4.2f s\n' % (epoch, epoch_loss / (train_set / batch_size), elapsed_time)
            
            
            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str)
            
            
            #loss
            Whole_Loss.append(epoch_L1_loss / (train_set / batch_size))
            if epoch % 5 == 0 or epoch == 1:
                torch.save(model, rf"{file_path}parameter/ours/{Model_name}/{Model_name}_lr4-4_bs8_Epoch{epoch}_2560_hybridLoss.pth")
                f.write(f"Model saved at epoch {epoch}\n")
                f.flush()
                print(f"Model saved at epoch {epoch}")


            if epoch % 100 == 0 or epoch == 1:
                #保存loss
                np.savetxt(f'{file_path}parameter/ours/Loss_{Model_name}_lr4-4_bs8_Epoch{epoch}_2560_hybridLoss.csv', Whole_Loss, delimiter=',')
'''
                #画出loss的散点图
                epoch = [i for i in range(1, Max_epoch+1)]
                plt.scatter(epoch, Whole_Loss)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                # plt.show()
                #保存loss的散点图
                plt.savefig('/data4/zhuo-file/fyc_file/ATTU_Norm/parameter/Ours/Loss_CTRM_UNetW4C_lr1-3_bs1_Epoch500_2560_hybridLoss.png')
'''
      