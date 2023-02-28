# coding:utf-8
import os
import argparse
from utils import *
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm
from time import time

def main(Method = 'SeAFusion', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    fusionmodel = FusionNet(output=1)
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(model_path))
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            start = time()
            fused_img = fusionmodel(vi_Y, img_ir)
            end = time()
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('{} | {} {:.4f}'.format(Method, img_name, end-start))
            
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
