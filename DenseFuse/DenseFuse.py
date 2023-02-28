# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

import time
from generate import generate
import os
import numpy as np
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import argparse


BATCH_SIZE = 2
EPOCHES = 4

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing.
# It is set as None when you want to train your own model.
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None
    
def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

num_epoch = 3

# time_save_name = '../Times_survey.xlsx'
# time_list = []
def main(Method = 'SeAFusion', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    # True for video sequences(frames)
    IS_VIDEO = False
    # True for RGB images
    
    ssim_weight = 10
    os.makedirs(save_dir, exist_ok=True)
    file_list = natsorted(os.listdir(vi_dir))
    test_bar = tqdm(file_list)
    for i, name in enumerate(test_bar):
        ir_name = os.path.join(ir_dir, name)
        vi_name = os.path.join(vi_dir, name)
        save_path = os.path.join(save_dir, name)
        fusion_type = 'addition'
        temp_time = generate(ir_name, vi_name, model_path, model_pre_path, ssim_weight, i+1, IS_VIDEO, IS_RGB=False, type=fusion_type, output_path=save_path, name=name)
        if is_RGB:
            img2RGB(save_path, vi_name)
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, name, temp_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default=True, help='fusion results dir')
    parser.add_argument('--is_RGB', type=bool, default=True, help='colorize fused images with visible color channels')
    opts = parser.parse_args()
    main(
        Method=opts.Method, 
        model_path=opts.model_path,
        ir_dir = opts.ir_dir,
        vi_dir = opts.vi_dir,
        save_dir = opts.save_dir,
        is_RGB=opts.is_RGB
    )