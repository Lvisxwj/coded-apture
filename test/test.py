from torch.autograd import Variable
import sys
sys.path.append("/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/")
import torch
from torch.utils.data import DataLoader
from Model import UNet
from dataset import dataset
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from train.loss import *
from train.Norm import max_min_norm
import scipy.io as sio

hsi_filename = r"/data4/zhuo-file/extracted_data/5nm_input/"
label_filename = r"/data4/zhuo-file/extracted_data/cor_r_label/"

# Load data and model
Dataset = dataset(hsi_filename, label_filename)

# GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.load(r"/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/parameter/ours/UNet/UNet_lr4-4_bs8_Epoch500_2560_hybridLoss.pth")
model.to(device)

# Hyperparameters
batch_size = 1
Max_epoch = 1
test_set = 1200

# Loss functions
loss_L1 = torch.nn.L1Loss()
loss_CharbonnierLoss = CharbonnierLoss() 
loss_SSIM = SSIM()

if __name__ == "__main__":
    log_file = r"./UNet_lr4-4_bs8_Epoch500_test_log.txt"
    Whole_Loss = []

    with open(log_file, "w") as f:
        for epoch in range(1, Max_epoch + 1):
            model.eval()
            test_loader = DataLoader(Dataset, num_workers=8, batch_size=batch_size, shuffle=False)

            # Initialize loss accumulators
            epoch_loss = 0.0 
            epoch_Charbonnier_loss = 0.0
            epoch_L1_loss = 0.0
            epoch_SSIM_loss = 0.0
            
            all_outputs = []
            all_labels = []
            
            start_time = time.time()

            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                print(labels.shape, outputs.shape)

                # Normalize outputs and labels
                # outputs = max_min_norm(outputs)
                labels = max_min_norm(labels)

                # Calculate losses
                Charbonnier_loss = loss_CharbonnierLoss(outputs, labels)
                L1_loss = loss_L1(outputs, labels)
                SSIM_loss = loss_SSIM(outputs, labels)
                loss = Charbonnier_loss + L1_loss + SSIM_loss
                
                epoch_loss += loss.item()
                epoch_Charbonnier_loss += Charbonnier_loss.item()
                epoch_L1_loss += L1_loss.item()
                epoch_SSIM_loss += SSIM_loss.item()
                
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                # Log results
                if i % (100) == 0:
                    log_str = '%4d %4d / %4d whole_loss = %.10f time = %s\n' % (epoch, i, test_set // batch_size, epoch_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d Charbonnierloss = %.10f time = %s\n' % (epoch, i, test_set // batch_size, epoch_Charbonnier_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d L1_loss = %.10f time = %s\n' % (epoch, i, test_set // batch_size, epoch_L1_loss / (i + 1),
                        datetime.datetime.now())
                    log_str += '%4d %4d / %4d SSIM_loss = %.10f time = %s\n' % (epoch, i, test_set // batch_size, epoch_SSIM_loss / (i + 1),
                        datetime.datetime.now())
                    
                    f.write(log_str)
                    f.flush()
                    print(log_str)

            elapsed_time = time.time() - start_time
            epoch_log_str = 'epoch = %4d , loss = %.10f , time = %4.2f s\n' % (epoch, epoch_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , Charbonnierloss = %.10f , time = %4.2f s\n' % (epoch, epoch_Charbonnier_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , L1_loss = %.10f , time = %4.2f s\n' % (epoch, epoch_L1_loss / (test_set / batch_size), elapsed_time)
            epoch_log_str += 'epoch = %4d , SSIM_loss = %.10f , time = %4.2f s\n' % (epoch, epoch_SSIM_loss / (test_set / batch_size), elapsed_time)
            
            f.write(epoch_log_str)
            f.flush()
            print(epoch_log_str)

            # Save loss for visualization
            Whole_Loss.append(epoch_L1_loss / (len(Dataset) / batch_size))
            
            sio.savemat(f'/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/result/UNet_Epoch{epoch}_results.mat', {
                'outputs': np.concatenate(all_outputs, axis=0),
                'labels': np.concatenate(all_labels, axis=0)
            })

    # Save loss data
    np.savetxt('./UNet_lr4-4_bs8_Epoch500_test_Loss.csv', Whole_Loss, delimiter=',')

    # Plot loss scatter
    epoch = [i for i in range(1, Max_epoch + 1)]
    plt.scatter(epoch, Whole_Loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # Save loss scatter plot
    plt.savefig('/data4/zhuo-file/fyc_file/Arti11/ATTU_Norm/result/UNet_lr4-4_bs8_Epoch500_test_Loss.png')
