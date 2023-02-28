import torch
import argparse
import os
from model import SuperFusion
from dataset import TestData, imsave
from time import time
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main(Method = 'SuperFusion', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    model = SuperFusion()
    model.resume(model_path)
    model = model.cuda()
    model.eval()
    test_dataloader = TestData(ir_dir, vi_dir)
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for idx, [ir, vi, name] in p_bar:
        vi_tensor = vi.cuda()
        ir_tenor = ir.cuda()
        start = time()
        with torch.no_grad():
            results = model.fusion_forward(ir_tenor, vi_tensor)
        end = time()
        imsave(results, os.path.join(save_dir, name))
        p_bar.set_description('{} | {} | {:.4f} s'.format(Method, name, end-start))
       
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