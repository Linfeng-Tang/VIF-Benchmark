# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import argparse
import glob
import os
from test_network import STDFusionNet
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
STDFusion_net = STDFusionNet()

class STDFusion:
    def __init__(self, Method='STDFusionNet', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True) -> None:
        self.Method = Method
        self.model_path = model_path
        self.ir_dir = ir_dir
        self.vi_dir = vi_dir
        self.save_dir = save_dir
        self.is_RGB = is_RGB
        self.reader = tf.compat.v1.train.NewCheckpointReader(model_path)
        os.makedirs(save_dir, exist_ok=True)
    def imread(self, path, is_grayscale=True):
        """
        Read image using its path.
        Default value  is gray-scale, and image is read by YCbCr format as the paper said.
        """
        if is_grayscale:
            # flatten=True Read the image as a grayscale map.
            return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
        else:
            return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

    def imsave(self, image, path):
        return scipy.misc.imsave(path, image)

    def prepare_data(self, dataset):
        self.data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(self.data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(self.data_dir, "*.bmp")))        
        data.extend(glob.glob(os.path.join(self.data_dir, "*.jpg")))
        natsorted(data)
        return data

    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    def input_setup(self, index):
        padding = 0
        sub_ir_sequence = []
        sub_vi_sequence = []
        input_ir = self.imread(self.data_ir[index]) / 127.5 - 1.0  # self.imread(self.data_ir[index]) / 255 #
        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])
        input_vi = self.imread(self.data_vi[index]) / 127.5 - 1.0  # (self.imread(self.data_vi[index]) - 127.5) / 127.5#
        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])
        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)
        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)
        return train_data_ir, train_data_vi, os.path.basename(self.data_ir[index])
    
    def img2RGB(self, f_name, vi_name):
        vi_img = Image.open(vi_name)
        vi_img = vi_img.convert('YCbCr')
        vi_Y, vi_Cb, vi_Cr = vi_img.split()
        f_img = Image.open(f_name).convert('L')
        f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
        f_RGB = f_img.convert('RGB')
        f_RGB.save(f_name)
        
    def STDFusion(self):
        with tf.name_scope('IR_input'):
            # infrared image patch
            ir_images = tf.placeholder(tf.float32, [1, None, None, 1], name='ir_images')
        with tf.name_scope('VI_input'):
            # visible image patch
            vi_images = tf.placeholder(tf.float32, [1, None, None, 1], name='vi_images')
        with tf.name_scope('fusion'):
            self.fusion_image, self.feature = STDFusion_net.STDFusion_model(vi_images, ir_images, self.reader)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            self.data_ir = self.prepare_data(self.ir_dir)
            self.data_vi = self.prepare_data(self.vi_dir)
            test_bar = tqdm(self.data_ir)
            for i, item in enumerate(test_bar):
                train_data_ir, train_data_vi, name = self.input_setup(i)
                start = time.time()
                result, encoding_feature = sess.run([self.fusion_image, self.feature], feed_dict={
                    ir_images: train_data_ir, vi_images: train_data_vi})
                result = result.squeeze()
                result = (result + 1) * 127.5
                end = time.time()
                save_path = os.path.join(self.save_dir, name)
                vi_path = os.path.join(self.vi_dir, name)
                self.imsave(result, save_path)
                if self.is_RGB:
                    self.img2RGB(save_path, vi_path)                
                test_bar.set_description('{} | {} | {:.4f} s'.format(self.Method, name, end-start))
                
def main(Method = 'STDFusionNet', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    test_STDFusion = STDFusion(Method=Method, 
        model_path=model_path,
        ir_dir = ir_dir,
        vi_dir = vi_dir,
        save_dir = save_dir,
        is_RGB=True )
    test_STDFusion.STDFusion()

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
