# -*- coding: utf-8 -*-
"""
@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Pytorch implement for "DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion" (IJCAI 2020)

https://www.ijcai.org/Proceedings/2020/135
"""

import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
from utils_didfuse import Test_fusion
from tqdm import tqdm
from natsort import natsorted
from time import time
import argparse
# =============================================================================
# Test Details 
# =============================================================================
def main(Method = 'DIDFUse', model_path_1='', model_path_2='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    
    device='cuda'
    addition_mode='l1_norm'#'Sum'&'Average'&'l1_norm'
    os.makedirs(save_dir, exist_ok=True)
    filelist = natsorted(os.listdir(ir_dir))
    test_bar = tqdm(filelist)
    for item in test_bar:
        ir_path = os.path.join(ir_dir, item)
        vi_path = os.path.join(vi_dir, item)
        save_path = os.path.join(save_dir, item)
        ir_img = Image.open(ir_path)
        vi_img = Image.open(vi_path)
        start = time()
        fused_img = Test_fusion(ir_img, vi_img, addition_mode, model_path_1, model_path_2)
        end = time()
        imsave(save_path,fused_img)
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, end-start))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path_1', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt', help='pretrained weights path')
    parser.add_argument('--model_path_2', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default=True, help='fusion results dir')
    parser.add_argument('--is_RGB', type=bool, default=True, help='colorize fused images with visible color channels')
    opts = parser.parse_args()
    main(
        Method=opts.Method, 
		model_path_1=opts.model_path_1,  
		model_path_2=opts.model_path_2,
        ir_dir = opts.ir_dir,
        vi_dir = opts.vi_dir,
        save_dir = opts.save_dir,
        is_RGB=opts.is_RGB
    )
