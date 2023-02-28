# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia

from DIDFuse import AE_Encoder,AE_Decoder

# =============================================================================
# Hyperparameters Setting 
# =============================================================================
Train_data_choose='FLIR'#'FLIR' & 'NIR'
if Train_data_choose=='FLIR':
    train_data_path = '.\\Datasets\\Train_data_FLIR\\'

root_VIS = train_data_path+'VIS\\'
root_IR = train_data_path+'\\IR\\'
train_path = '.\\Train_result\\'
device = "cuda"

batch_size=24
channel=64
epochs = 120
lr = 1e-3

Train_Image_Number=len(os.listdir(train_data_path+'IR\\IR'))

Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size
# =============================================================================
# Preprocessing and dataset establishment 
# =============================================================================

transforms = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])

                                   
Data_VIS = torchvision.datasets.ImageFolder(root_VIS,transform=transforms)
dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, batch_size,shuffle=False)

Data_IR = torchvision.datasets.ImageFolder(root_IR,transform=transforms)
dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size,shuffle=False)

# =============================================================================
# Models
# =============================================================================
AE_Encoder=AE_Encoder()
AE_Decoder=AE_Decoder()
is_cuda = True
if is_cuda:
    AE_Encoder=AE_Encoder.cuda()
    AE_Decoder=AE_Decoder.cuda()
 
 
print(AE_Encoder)
print(AE_Decoder)


optimizer1 = optim.Adam(AE_Encoder.parameters(), lr = lr)
optimizer2 = optim.Adam(AE_Decoder.parameters(), lr = lr)


scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [epochs//3,epochs//3*2], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [epochs//3,epochs//3*2], gamma=0.1)


MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
ssim = kornia.losses.SSIM(11, reduction='mean')
# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
loss_train=[]
mse_loss_B_train=[]
mse_loss_D_train=[]
mse_loss_VF_train=[]
mse_loss_IF_train=[]
Gradient_loss_train=[]
lr_list1=[]
lr_list2=[]
alpha_list=[]
for iteration in range(epochs):
    
    AE_Encoder.train()
    AE_Decoder.train()
    
   
    data_iter_VIS = iter(dataloader_VIS)
    data_iter_IR = iter(dataloader_IR)
    
    for step in range(Iter_per_epoch):
        data_VIS,_ =next(data_iter_VIS)
        data_IR,_  =next(data_iter_IR)
        
          
        if is_cuda:
            data_VIS=data_VIS.cuda()
            data_IR=data_IR.cuda()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # =====================================================================
        # Calculate loss 
        # =====================================================================
        feature_V_1,feature_V_2,feature_V_B, feature_V_D = AE_Encoder(data_VIS)
        feature_I_1,feature_I_2,feature_I_B, feature_I_D = AE_Encoder(data_IR)
        img_recon_V=AE_Decoder(feature_V_1,feature_V_2,feature_V_B, feature_V_D)
        img_recon_I=AE_Decoder(feature_I_1,feature_I_2,feature_I_B, feature_I_D)

        mse_loss_B  = L1Loss(feature_I_B, feature_V_B)
        mse_loss_D  = L1Loss(feature_I_D, feature_V_D)

        mse_loss_VF = 5*ssim(data_VIS, img_recon_V)+MSELoss(data_VIS, img_recon_V)
        mse_loss_IF = 5*ssim(data_IR,  img_recon_I)+MSELoss(data_IR,  img_recon_I)

        Gradient_loss = L1Loss(
                kornia.filters.SpatialGradient()(data_VIS),
                kornia.filters.SpatialGradient()(img_recon_V)
                )
        #Total loss
        loss = 2*mse_loss_VF + 2*mse_loss_IF +  torch.tanh(mse_loss_B) - 0.5*torch.tanh(mse_loss_D) + 10*Gradient_loss
   
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        
        los = loss.item()
        los_B = mse_loss_B.item()
        los_D = mse_loss_D.item()
        los_VF = mse_loss_VF.item()
        los_IF = mse_loss_IF.item()
        los_G = Gradient_loss.item()
        
        print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, step+1, los, optimizer1.state_dict()['param_groups'][0]['lr']))

        #Save Loss
        loss_train.append(loss.item())
        mse_loss_B_train.append(mse_loss_B.item())
        mse_loss_D_train.append(mse_loss_D.item())
        mse_loss_VF_train.append(mse_loss_VF.item())
        mse_loss_IF_train.append(mse_loss_IF.item())
        Gradient_loss_train.append(Gradient_loss.item())
    scheduler1.step()
    scheduler2.step()
    lr_list1.append(optimizer1.state_dict()['param_groups'][0]['lr'])
    lr_list2.append(optimizer2.state_dict()['param_groups'][0]['lr'])


# Save Weights and result
torch.save( {'weight': AE_Encoder.state_dict(), 'epoch':epochs}, 
   os.path.join(train_path,'Encoder_weight.pkl'))
torch.save( {'weight': AE_Decoder.state_dict(), 'epoch':epochs}, 
   os.path.join(train_path,'Decoder_weight.pkl'))

scio.savemat(os.path.join(train_path, 'TrainData.mat'), 
                         {'Loss': np.array(loss_train),
                          'Base_layer_loss'  : np.array(mse_loss_B_train),
                          'Detail_layer_loss': np.array(mse_loss_D_train),
                          'V_recon_loss': np.array(mse_loss_VF_train),
                          'I_recon_loss': np.array(mse_loss_IF_train),
                          'Gradient_loss': np.array(Gradient_loss_train),
                          })
scio.savemat(os.path.join(train_path, 'TrainData_plot_loss.mat'), 
                         {'loss_train': np.array(loss_train),
                          'mse_loss_B_train'  : np.array(mse_loss_B_train),
                          'mse_loss_D_train': np.array(mse_loss_D_train),
                          'mse_loss_VF_train': np.array(mse_loss_VF_train),
                          'mse_loss_IF_train': np.array(mse_loss_IF_train),
                          'Gradient_loss_train': np.array(Gradient_loss_train),
                          })
# plot
def Average_loss(loss):
    return [sum(loss[i*Iter_per_epoch:(i+1)*Iter_per_epoch])/Iter_per_epoch for i in range(int(len(loss)/Iter_per_epoch))]

plt.figure(figsize=[12,8])
plt.subplot(2,3,1), plt.plot(Average_loss(loss_train)), plt.title('Loss')
plt.subplot(2,3,2), plt.plot(Average_loss(mse_loss_B_train)), plt.title('Base_layer_loss')
plt.subplot(2,3,3), plt.plot(Average_loss(mse_loss_D_train)), plt.title('Detail_layer_loss')
plt.subplot(2,3,4), plt.plot(Average_loss(mse_loss_VF_train)), plt.title('V_recon_loss')
plt.subplot(2,3,5), plt.plot(Average_loss(mse_loss_IF_train)), plt.title('I_recon_loss')
plt.subplot(2,3,6), plt.plot(Average_loss(Gradient_loss_train)), plt.title('Gradient_loss')
plt.tight_layout()
plt.savefig(os.path.join(train_path,'curve_per_epoch.png'))    


