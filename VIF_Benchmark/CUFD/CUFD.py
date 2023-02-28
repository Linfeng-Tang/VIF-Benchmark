# Demo - train the DenseFuse network & use it to generate an image
from __future__ import print_function
from generate import generate
import time
import os
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import argparse

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)
    
def main(Method = 'CUFD', model_path_1='', model_path_2='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):
    filelist = natsorted(os.listdir(ir_dir))
    test_bar = tqdm(filelist)
    os.makedirs(save_dir, exist_ok=True)
    for i, item in enumerate(test_bar):
        infrared = os.path.join(ir_dir, item)
        visible = os.path.join(vi_dir, item)
        save_path = os.path.join(save_dir, item)
        start = time.time()
        generate(infrared, visible, model_path_1, model_path_2, output_path=save_path)
        end = time.time()
        if is_RGB:
            img2RGB(save_path, visible)            
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, end-start))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path_1', type=str, default='/data1/timer/Fusion_Benchmark/Checkpoint/CUFD/1part1_model.ckpt', help='pretrained weights path')
    parser.add_argument('--model_path_2', type=str, default='/data1/timer/Fusion_Benchmark/Checkpoint/CUFD/part2_model.ckpt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default='/data1/timer/Fusion_Benchmark/Results/CUFD', help='fusion results dir')
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
