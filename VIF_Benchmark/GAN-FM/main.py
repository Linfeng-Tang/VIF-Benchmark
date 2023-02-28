#coding:utf-8
from __future__ import print_function
import time
import h5py
import numpy as np
from train import train
from generate import generate
import os 
import glob

BATCH_SIZE =16
EPOCHES = 20

LOGGING = 50
MODEL_SAVE_PATH = './model/'
IS_TRAINING = False


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
    data.sort()
    filenames.sort()
    return data, filenames


def main():
	if IS_TRAINING:
		f = h5py.File('./train.h5', 'r')
		sources = f['data'][:]
		sources = np.transpose(sources, (0, 2, 3, 1))
		print(('\nBegin to train the network ...\n'))
		train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)

	else:
		print('\nBegin to generate pictures ...\n')
		Time=[]
		model_path = MODEL_SAVE_PATH+'model.ckpt'
		save_path='./results/'
		# os.makedirs(save_path)
		ir_paths,ir_names=prepare_data_path(r'./test_imgs/ir')
		vis_paths,vis_names=prepare_data_path(r'./test_imgs/vis')
		for i in range(len(ir_paths)):
			ir_path = ir_paths[i]
			vis_path = vis_paths[i]
			begin = time.time()
			generate(ir_path, vis_path, model_path,ir_names[i], output_path= save_path)
			end = time.time()
			Time.append(end - begin)
			print(ir_names[i])
		print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))

if __name__ == '__main__':
	main()
