# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""
import numpy as np
import torch
from Network import AE_Encoder,AE_Decoder
import torch.nn.functional as F

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

device='cuda'

def output_img(x):
    return (x.cpu().detach().numpy()[0,:,:,:].transpose(1, 2, 0)*255.0).astype('uint8')

def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def Test_fusion(img_test1,img_test2,addition_mode='Sum', 
                En_model_path="/data/timer/Comparison/VIF/DIDFuse/Models/Encoder_weight_IJCAI.pkl", 
                De_model_path="/data/timer/Comparison/VIF/DIDFuse/Models/Decoder_weight_IJCAI.pkl"):
    AE_Encoder1 = AE_Encoder().to(device)
    AE_Encoder1.load_state_dict(torch.load(En_model_path)['weight'])
    
    AE_Decoder1 = AE_Decoder().to(device)
    AE_Decoder1.load_state_dict(torch.load(De_model_path)['weight'])
    AE_Encoder1.eval()
    AE_Decoder1.eval()
    
    img_test1 = np.array(img_test1, dtype='float32')/255# 将其转换为一个矩阵
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    ## 假设img_test2是彩色图像，使用其Y通道进行融合
    img_test2 = np.array(img_test2, dtype='float32')/255 # 将其转换为一个矩阵
    img_test2 =  np.expand_dims(img_test2.transpose(2, 0, 1), axis=0)
    img_test2 = torch.from_numpy(img_test2)
    
    vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_test2)
    
    img_test1=img_test1.cuda()
    img_test2=vi_Y.cuda()    
    vi_Cb = vi_Cb.cuda()
    vi_Cr = vi_Cr.cuda()
    
    with torch.no_grad():
        F_i1,F_i2,F_ib,F_id=AE_Encoder1(img_test1)
        F_v1,F_v2,F_vb,F_vd=AE_Encoder1(img_test2)
        
    if addition_mode=='Sum':      
        F_b=(F_ib+F_vb)
        F_d=(F_id+F_vd)
        F_1=(F_i1+F_v1)
        F_2=(F_i2+F_v2)
    elif addition_mode=='Average':
        F_b=(F_ib+F_vb)/2         
        F_d=(F_id+F_vd)/2
        F_1=(F_i1+F_v1)/2
        F_2=(F_i2+F_v2)/2
    elif addition_mode=='l1_norm':
        F_b=l1_addition(F_ib,F_vb)
        F_d=l1_addition(F_id,F_vd)
        F_1=l1_addition(F_i1,F_v1)
        F_2=l1_addition(F_i2,F_v2)
    else:
        print('Wrong!')
         
    with torch.no_grad():
        Out = AE_Decoder1(F_1,F_2,F_b,F_d)
    
    Out = YCbCr2RGB(Out, vi_Cb, vi_Cr)
    return output_img(Out)