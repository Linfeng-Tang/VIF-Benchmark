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
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))    
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    natsorted(data)
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_ir, img_vi, reader):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/w1_vi')))
            bias=tf.get_variable("b1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/b1_vi')))
            conv1_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_vi = lrelu(conv1_vi)            
                      
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/w2_vi')))
            bias=tf.get_variable("b2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/b2_vi')))
            conv2_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_vi = lrelu(conv2_vi)            
         
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)            
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/w3_vi')))
            bias=tf.get_variable("b3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/b3_vi')))
            conv3_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_vi, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_vi = lrelu(conv3_vi)            
          
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_ir, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
            
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/w4_vi')))
            bias=tf.get_variable("b4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/b4_vi')))
            conv4_vi= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_vi, weights, strides=[1,1,1,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_vi = lrelu(conv4_vi)            
            
        conv_ir_vi =tf.concat([conv4_ir,conv4_vi,conv4_ir,conv4_vi],axis=-1)            
                     
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_ir= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

def input_setup(data_ir, data_vi, index):
    padding=6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi, os.path.basename(data_ir[index])

def main(Method = 'GANMcC', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True): 
    reader = tf.train.NewCheckpointReader(model_path)
    
    os.makedirs(save_dir, exist_ok=True) 
    with tf.name_scope('IR_input'):
        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.name_scope('VI_input'):
        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
    with tf.name_scope('input'):
        input_image_ir =tf.concat([images_ir,images_ir,images_vi],axis=-1)
        input_image_vi =tf.concat([images_vi,images_vi,images_ir],axis=-1)

    with tf.name_scope('fusion'):
        fusion_image=fusion_model(input_image_ir, input_image_vi, reader)


    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        data_ir=prepare_data(ir_dir)
        data_vi=prepare_data(vi_dir)
        test_bar = tqdm(data_ir)
        for i, data in enumerate(test_bar):
            
            start=time.time()
            ir_img, vi_img, name=input_setup(data_ir, data_vi, i)
            result =sess.run(fusion_image,feed_dict={images_ir: ir_img, images_vi: vi_img})
            result=result*127.5+127.5
            result = result.squeeze()
            save_path = os.path.join(save_dir, name)
            vi_path = os.path.join(vi_dir, name)
            end=time.time()
            # print(out.shape)
            imsave(result, save_path)
            if is_RGB:
                img2RGB(save_path, vi_path)            
            test_bar.set_description('{} | {} | {:.4f} s'.format(Method, name, end-start))
            
            
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

