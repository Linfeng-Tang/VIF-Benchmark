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
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    natsorted(data)
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img_ir, img_vi, reader):
    with tf.variable_scope('fusion_model'):
    ####################  Layer1  ###########################
        with tf.variable_scope('layer1_ir'):
            weights=tf.get_variable("w1_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_ir/w1_ir')))
            bias=tf.get_variable("b1_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_ir/b1_ir')))
            conv1_ir= tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_ir = lrelu(conv1_ir)   
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/w1_vi')))
            bias=tf.get_variable("b1_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer1_vi/b1_vi')))
            conv1_vi= tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_vi = lrelu(conv1_vi)           
            
####################  Layer2  ###########################            
        with tf.variable_scope('layer2_ir'):
            weights=tf.get_variable("w2_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_ir/w2_ir')))
            bias=tf.get_variable("b2_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_ir/b2_ir')))
            conv2_ir= tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_ir = lrelu(conv2_ir)         
            
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/w2_vi')))
            bias=tf.get_variable("b2_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer2_vi/b2_vi')))
            conv2_vi= tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_vi = lrelu(conv2_vi)            
                                  
####################  Layer3  ###########################               
        conv_12_ir=tf.concat([conv1_ir,conv2_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi],axis=-1)        
            
        with tf.variable_scope('layer3_ir'):
            weights=tf.get_variable("w3_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_ir/w3_ir')))
            bias=tf.get_variable("b3_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_ir/b3_ir')))
            conv3_ir= tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_ir =lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/w3_vi')))
            bias=tf.get_variable("b3_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer3_vi/b3_vi')))
            conv3_vi= tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_vi = lrelu(conv3_vi)
            

####################  Layer4  ########################### 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi],axis=-1)                   
            
        with tf.variable_scope('layer4_ir'):
            weights=tf.get_variable("w4_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_ir/w4_ir')))
            bias=tf.get_variable("b4_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_ir/b4_ir')))
            conv4_ir= tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/w4_vi')))
            bias=tf.get_variable("b4_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer4_vi/b4_vi')))
            conv4_vi= tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_vi = lrelu(conv4_vi)
            
 
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
####################  Layer5  ###########################         
        with tf.variable_scope('layer5_fuse'):
            weights=tf.get_variable("w5_fuse",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_fuse/w5_fuse')))
            bias=tf.get_variable("b5_fuse",initializer=tf.constant(reader.get_tensor('fusion_model/layer5_fuse/b5_fuse')))
            conv5_fuse= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_fuse=tf.nn.tanh(conv5_fuse)
            
####################  Layer6  ########################### 
        with tf.variable_scope('layer6_sept'):
            weights=tf.get_variable("w6_sept",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_sept/w6_sept')))
            bias=tf.get_variable("b6_sept",initializer=tf.constant(reader.get_tensor('fusion_model/layer6_sept/b6_sept')))
            conv6_sept= tf.nn.conv2d(conv5_fuse, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_sept=lrelu(conv6_sept)
            
####################  Layer7  ###########################            
        with tf.variable_scope('layer7_ir'):
            weights=tf.get_variable("w7_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_ir/w7_ir')))
            bias=tf.get_variable("b7_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_ir/b7_ir')))
            conv7_ir= tf.nn.conv2d(conv6_sept, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_ir = lrelu(conv7_ir)         
            
        with tf.variable_scope('layer7_vi'):
            weights=tf.get_variable("w7_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_vi/w7_vi')))
            bias=tf.get_variable("b7_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer7_vi/b7_vi')))
            conv7_vi= tf.nn.conv2d(conv6_sept, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_vi = lrelu(conv7_vi) 

####################  Layer8  ###########################            
        with tf.variable_scope('layer8_ir'):
            weights=tf.get_variable("w8_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_ir/w8_ir')))
            bias=tf.get_variable("b8_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_ir/b8_ir')))
            conv8_ir= tf.nn.conv2d(conv7_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_ir = lrelu(conv8_ir)         
            
        with tf.variable_scope('layer8_vi'):
            weights=tf.get_variable("w8_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_vi/w8_vi')))
            bias=tf.get_variable("b8_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer8_vi/b8_vi')))
            conv8_vi= tf.nn.conv2d(conv7_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_vi = lrelu(conv8_vi)
             
####################  Layer9  ###########################            
        with tf.variable_scope('layer9_ir'):
            weights=tf.get_variable("w9_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer9_ir/w9_ir')))
            bias=tf.get_variable("b9_ir",initializer=tf.constant(reader.get_tensor('fusion_model/layer9_ir/b9_ir')))
            conv9_ir = tf.nn.conv2d(conv8_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9_ir = tf.nn.tanh(conv9_ir)         
            
        with tf.variable_scope('layer9_vi'):
            weights=tf.get_variable("w9_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer9_vi/w9_vi')))
            bias=tf.get_variable("b9_vi",initializer=tf.constant(reader.get_tensor('fusion_model/layer9_vi/b9_vi')))
            conv9_vi= tf.nn.conv2d(conv8_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9_vi = tf.nn.tanh(conv9_vi)            
            
    return conv5_fuse,conv9_ir,conv9_vi
           

def input_setup(data_ir, data_vi, index):
    padding=0
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
    name_ir = os.path.basename(data_ir[index])
    name_vi = os.path.basename(data_vi[index])
    return train_data_ir, train_data_vi, name_ir, name_vi

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    return flops.total_float_ops, params.total_parameters

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

def main(Method = 'SDNet', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    reader = tf.train.NewCheckpointReader(model_path)
    A_path = os.path.join(ir_dir)
    B_path = os.path.join(vi_dir)
    os.makedirs(save_dir, exist_ok=True)
    with tf.name_scope('IR_input'):
        images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    with tf.name_scope('VI_input'):
        images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
          
    with tf.name_scope('input'):
        input_image_ir =images_ir
        input_image_vi =images_vi

    with tf.name_scope('fusion'):
        fusion_image, sept_ir, sept_vi = fusion_model(input_image_ir,input_image_vi, reader)


    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        data_ir = prepare_data(A_path)
        data_vi = prepare_data(B_path)
        test_bar = tqdm(data_ir)
        for i, data in enumerate(test_bar):
            train_data_ir,train_data_vi, name_ir, name_vi = input_setup(data_ir, data_vi, i)
            start=time.time()
            result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi})
            result=result*127.5+127.5
            result = result.squeeze()
            save_path = os.path.join(save_dir, name_ir)
            vi_path = os.path.join(vi_dir, name_ir)
            end = time.time()
            # print(out.shape)
            imsave(result, save_path)
            if is_RGB:
                img2RGB(save_path, vi_path)            
            test_bar.set_description('{} | {} | {:.4f} s'.format(Method, name_ir, end-start))
            
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

  