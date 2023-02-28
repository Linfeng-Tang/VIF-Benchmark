#coding:utf-8
from __future__ import print_function
import time
from generate_GAN_FM import generate
import os
import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import argparse


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    # os.getcwd()
    # data_dir = os.path.join(os.getcwd(), dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    # data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    natsorted(data)
    natsorted(filenames)
    return data, filenames

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)
    
def main(Method = 'GAN-FM', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    ir_paths, ir_names=prepare_data_path(ir_dir)
    vis_paths, vis_names=prepare_data_path(vi_dir)
    test_bar = tqdm(ir_paths)
    for i, data in enumerate(test_bar):
        ir_path = ir_paths[i]
        vis_path = vis_paths[i]
        name = os.path.basename(ir_path)
        save_path = os.path.join(save_dir, name)
        start = time.time()
        generate(ir_path, vis_path, model_path, output_path= save_path)
        end = time.time()
        if is_RGB:
            img2RGB(save_path, vis_path)            
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, name, end-start))
            

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

