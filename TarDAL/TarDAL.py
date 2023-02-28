import argparse
from argparse import Namespace

import torch
from pathlib import Path

from modules.generator import Generator
from pipeline.eval import Eval
from natsort import natsorted
import os


def img_filter(x: Path) -> bool:
    return x.suffix in ['.png', '.bmp', '.jpg']

def main(Method = 'TarDAL', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  

    # init model
    net = Generator(dim=32, depth=3)

    # load pretrained weights
    ck_pt = torch.load(model_path)
    net.load_state_dict(ck_pt)

    # images
    root_ir = Path(ir_dir)
    root_vi = Path(vi_dir)
    ir_paths = [x for x in natsorted((root_ir).glob('*')) if img_filter]
    vi_paths = [x for x in natsorted((root_vi).glob('*')) if img_filter]

    # fuse
    f = Eval(net, cudnn=True, half=True, eval=True)

    f(ir_paths, vi_paths, Path(save_dir), True)

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