# -*- coding:utf-8 -*-
# @Project: NestFuse for image fusion
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @File : test.py

import os
import torch
from torch.autograd import Variable
from net import NestFuse_autoencoder
import utils_NestFuse as utils_NestFuse
from args_fusion import args
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
from time import time
import argparse


def load_model(path, deepsupervision=False):
    input_nc = 1
    output_nc = 1
    nb_filter = [64, 112, 160, 208, 256]

    nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
    nest_model.load_state_dict(torch.load(path))
    nest_model.eval()
    nest_model.cuda()

    return nest_model


def run_demo(nest_model, infrared_path, visible_path, output_path, f_type):
    img_ir, h, w, c = utils_NestFuse.get_test_image(infrared_path)
    img_vi, h, w, c = utils_NestFuse.get_test_image(visible_path)

    # dim = img_ir.shape
    if c is 1:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        # encoder
        en_r = nest_model.encoder(img_ir)
        en_v = nest_model.encoder(img_vi)
        # fusion
        f = nest_model.fusion(en_r, en_v, f_type)
        # decoder
        img_fusion_list = nest_model.decoder_eval(f)
    else:
        # fusion each block
        img_fusion_blocks = []
        for i in range(c):
            # encoder
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)

            en_r = nest_model.encoder(img_ir_temp)
            en_v = nest_model.encoder(img_vi_temp)
            # fusion
            f = nest_model.fusion(en_r, en_v, f_type)
            # decoder
            img_fusion_temp = nest_model.decoder_eval(f)
            img_fusion_blocks.append(img_fusion_temp)
        img_fusion_list = utils_NestFuse.recons_fusion_images(img_fusion_blocks, h, w)

    ############################ multi outputs ##############################################
    for img_fusion in img_fusion_list:
        utils_NestFuse.save_image_test(img_fusion, output_path)


def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)


def main(Method='NestFuse', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):
    # run demo
    deepsupervision = False  # true for deeply supervision
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        model = load_model(model_path, deepsupervision)
        filelist = natsorted(os.listdir(ir_dir))

        f_type = 'attention_max'
        test_bar = tqdm(filelist)

        for i, item in enumerate(test_bar):
            index = i + 1
            infrared_path = os.path.join(ir_dir, item)
            visible_path = os.path.join(vi_dir, item)
            save_path = os.path.join(save_dir, item)
            start = time()
            run_demo(model, infrared_path, visible_path, save_path, f_type)
            end = time()
            if is_RGB:
                img2RGB(save_path, visible_path)
            test_bar.set_description('{} | {} {:.4f}'.format(Method, item, end - start))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt',
                        help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir',
                        help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi',
                        help='visible image dir')
    parser.add_argument('--save_dir', type=str, default='', help='fusion results dir')
    parser.add_argument('--is_RGB', type=bool, default=True, help='colorize fused images with visible color channels')
    opts = parser.parse_args()
    main(
        Method=opts.Method,
        model_path=opts.model_path,
        ir_dir=opts.ir_dir,
        vi_dir=opts.vi_dir,
        save_dir=opts.save_dir,
        is_RGB=opts.is_RGB
    )


