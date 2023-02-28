# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: hui_li_jnu@163.com
# @File : test_40pairs.py
# @Time : 2020/8/14 17:11

# test phase
import os
import torch
from torch.autograd import Variable
from net import NestFuse_light2_nodense, Fusion_network, Fusion_strategy
import utils_RFN_Nest
from args_fusion import args
import numpy as np
import  time
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)


def load_model(path_auto, path_fusion, fs_type, flag_img):
	if flag_img is True:
		nc = 3
	else:
		nc =1
	input_nc = nc
	output_nc = nc
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
	nest_model.load_state_dict(torch.load(path_auto))

	fusion_model = Fusion_network(nb_filter, fs_type)
	fusion_model.load_state_dict(torch.load(path_fusion))

	fusion_strategy = Fusion_strategy(fs_type)

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	fusion_model.eval()
	nest_model.cuda()
	fusion_model.cuda()

	return nest_model, fusion_model, fusion_strategy


def run_demo(nest_model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type, use_strategy, flag_img):
	img_ir, h, w, c = utils_RFN_Nest.get_test_image(infrared_path, flag=flag_img)  # True for rgb
	img_vi, h, w, c = utils_RFN_Nest.get_test_image(visible_path, flag=flag_img)

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
		# fusion net
		if use_strategy:
			f = fusion_strategy(en_r, en_v)
		else:
			f = fusion_model(en_r, en_v)
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
			# fusion net
			if use_strategy:
				f = fusion_strategy(en_r, en_v)
			else:
				f = fusion_model(en_r, en_v)
			# decoder
			img_fusion_temp = nest_model.decoder_eval(f)
			img_fusion_blocks.append(img_fusion_temp)
		img_fusion_list = utils_RFN_Nest.recons_fusion_images(img_fusion_blocks, h, w)

	# ########################### multi-outputs ##############################################
	for img_fusion in img_fusion_list:
		utils_RFN_Nest.save_image_test(img_fusion, output_path_root)



def main(Method = 'RFN-Nest', model_path_1='', model_path_2='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
	# False - gray
	flag_img = False	
	fs_type = 'res'  # res (RFN), add, avg, max, spa, nuclear
	use_strategy = False  # True - static strategy; False - RFN
	with torch.no_grad():
		model, fusion_model, fusion_strategy = load_model(model_path_2, model_path_1, fs_type, flag_img)	
		os.makedirs(save_dir, exist_ok=True)
		file_list = natsorted(os.listdir(ir_dir))
		test_bar = tqdm(file_list)
		for item in test_bar:
			infrared_path = os.path.join(os.path.abspath(ir_dir), item)
			visible_path = os.path.join(os.path.abspath(vi_dir), item)
			output_path = os.path.join(os.path.abspath(save_dir), item)
			begin_time = time.time()
			name_ir = item
			run_demo(model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path, name_ir, fs_type, use_strategy, flag_img)
			end_time = time.time()
			proc_time = end_time - begin_time
			if is_RGB:
				img2RGB(output_path, visible_path)                
			test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, proc_time))
                



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