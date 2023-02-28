
import os
import cv2
import time
import torch
from model_IFCNN import myIFCNN
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils.myTransforms import denorm
import argparse
from natsort import natsorted

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)
    
def main(Method = 'SeAFusion', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    fuse_scheme = 0
    model = myIFCNN(fuse_scheme=fuse_scheme)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()
    os.makedirs(save_dir, exist_ok=True)
    file_list = natsorted(os.listdir(ir_dir))
    mean=[0.485, 0.456, 0.406] # normalization parameters
    std=[0.229, 0.224, 0.225]
    test_bar = tqdm(file_list)
    for i, item in enumerate(test_bar):
        if item.endswith('.bmp') or item.endswith('.png') or item.endswith('.jpg') or item.endswith('.tif'):
            ir_image_name = os.path.join(ir_dir, item)
            vi_image_name = os.path.join(vi_dir, item)
            fused_image_name = os.path.join(save_dir, item)
            img1 = cv2.imread(ir_image_name, 0)
            [h, w] = img1.shape
            img2 = cv2.imread(vi_image_name, 0)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            transform1 = transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])

            img1 = transform1(img1)
            img2 = transform1(img2)
            img1.unsqueeze_(0)
            img2.unsqueeze_(0)
            begin_time = time.time()
            # perform image fusion
            with torch.no_grad():
                res = model(Variable(img1.cuda()), Variable(img2.cuda()))
                res = denorm(mean, std, res[0]).clamp(0, 1) * 255
                res_img = res.cpu().data.numpy().astype('uint8')
                img = res_img.transpose([1,2,0])
            proc_time = time.time() - begin_time
            img = cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(img)
            img.save(fused_image_name)
            if is_RGB:
                img2RGB(fused_image_name, vi_image_name)  
            test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, proc_time))
    torch.cuda.empty_cache()
                
           

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
    



