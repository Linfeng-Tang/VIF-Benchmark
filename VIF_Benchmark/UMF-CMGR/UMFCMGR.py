import sys
import pathlib
from PIL import Image
import time

import os
import cv2
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import torch
from dataloader.fuse_data_vsm import FuseTestData
from models.fusion_net import FusionNet
import argparse

def main(Method='UMF-CMGR', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    cuda = True
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True
    
    data = FuseTestData(ir_dir, vi_dir)
    test_data_loader = torch.utils.data.DataLoader(data, 1, False, pin_memory=True)
    net = FusionNet(nfeats=64).to(device)

    print("===> loading trained model '{}'".format(model_path))
    model_state_dict = torch.load(model_path)['net']
    net.load_state_dict(model_state_dict)

    print("===> Starting Testing")    
    os.makedirs(save_dir, exist_ok=True)
    net.eval()
    test_bar = tqdm(test_data_loader)
    for (ir, vi), (ir_path, vi_path) in test_bar:
        file_name = os.path.basename(ir_path[0])
        ir = ir.cuda()
        vi = vi.cuda()
        start = time.time()
        with torch.no_grad():
            fuse_out  = net(ir, vi)
        end = time.time()        
        save_name = os.path.join(save_dir, file_name)
        imsave(fuse_out, save_name)
        if is_RGB:
            img2RGB(save_name, vi_path[0])                
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, file_name, end-start))
                


def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)
    
def test(net, test_data_loader, dst, device):
    pass
        
def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    # dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        # p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(dst, im_cv)


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