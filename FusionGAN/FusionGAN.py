# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        # flatten=True 以灰度图的形式读取
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    # data.sort(key=lambda x: int(x[len(data_dir) + 1:-5]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fusion_model(img, reader=None):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ir = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ir = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias = tf.get_variable("b4", initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ir = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights = tf.get_variable("w5", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias = tf.get_variable("b5", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir = tf.nn.conv2d(conv4_ir, weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            conv5_ir = tf.nn.tanh(conv5_ir)
    return conv5_ir

def load_test_data(image_path, mode=1):
    padding = 6
    img = imread(image_path)
    img  = np.lib.pad(img, ((padding, padding), (padding, padding)), 'edge')
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = preprocessing(img)
    return img

def preprocessing(x):
    x = (x  - 127.5) / 127.5 # -1 ~ 1
    return x

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

def main(Method = 'FusionGAN', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    file_list = natsorted(os.listdir(ir_dir))
    reader = tf.train.NewCheckpointReader(model_path)
    
    with tf.name_scope('IR_input'):
        # 红外图像patch
        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
        # images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.name_scope('VI_input'):
        # 可见光图像patch
        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
        # self.labels_vi_gradient=gradient(self.labels_vi)
    # 将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        # resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
        input_image = tf.concat([images_ir, images_vi], axis=-1)
    with tf.name_scope('fusion'):
        fusion_image = fusion_model(input_image, reader)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        test_bar = tqdm(file_list)
        for i, item in enumerate(test_bar):
            if item.endswith('.bmp') or item.endswith('.png') or item.endswith('.jpg'):
                ir_image_name = os.path.join(os.path.abspath(ir_dir), item)
                vi_image_name = os.path.join(os.path.abspath(vi_dir), item)
                fused_image_name = os.path.join(os.path.abspath(save_dir), item)
                train_data_ir = load_test_data(ir_image_name)
                train_data_vi = load_test_data((vi_image_name))
                start = time.time()
                result = sess.run(fusion_image, feed_dict={images_ir: train_data_ir, images_vi: train_data_vi})
                result = result * 127.5 + 127.5
                result = result.squeeze()
                end = time.time()
                imsave(result, fused_image_name)
                if is_RGB:
                    img2RGB(fused_image_name, vi_image_name)                
                test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, end-start))
                
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
