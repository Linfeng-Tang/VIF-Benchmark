# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  l2_norm,
  blur_2th,
  low_pass,
  tf_ssim
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

class Fusion(object):

  def __init__(self, 
               sess, 
               image_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
    with tf.name_scope('VI_input'):
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')


    with tf.name_scope('input'):
        self.input_image_ir =self.images_ir
        self.input_image_vi =self.images_vi


    with tf.name_scope('fusion'): 
        self.fusion_image,self.sept_ir,self.sept_vi=self.fusion_model(self.input_image_ir,self.input_image_vi)

    with tf.name_scope('grad_bin'):
        self.Image_vi_grad=tf.abs(gradient(self.images_vi))
        self.Image_ir_grad=tf.abs(gradient(self.images_ir))
        
        self.Image_vi_grad_lowpass=tf.abs(gradient(low_pass(self.images_vi)))
        self.Image_ir_grad_lowpass=tf.abs(gradient(low_pass(self.images_ir)))
        
        self.Image_vi_weight=self.Image_vi_grad
        self.Image_ir_weight=self.Image_ir_grad
        
        self.Image_vi_weight_lowpass=self.Image_vi_grad_lowpass
        self.Image_ir_weight_lowpass=self.Image_ir_grad_lowpass
        
        self.Image_vi_score_1=1
        self.Image_ir_score_1=1         
        self.Image_vi_score_2=tf.sign(self.Image_vi_weight_lowpass-tf.minimum(self.Image_vi_weight_lowpass,self.Image_ir_weight_lowpass))
        self.Image_ir_score_2=tf.sign(self.Image_ir_weight_lowpass-tf.minimum(self.Image_vi_weight_lowpass,self.Image_ir_weight_lowpass))        
        self.Image_vi_score=self.Image_vi_score_1*self.Image_vi_score_2
        self.Image_ir_score=1-self.Image_vi_score
        


    with tf.name_scope('g_loss'):
        self.g_loss_int=tf.reduce_mean(tf.square(self.fusion_image - self.images_ir))+0.5*tf.reduce_mean(tf.square(self.fusion_image - self.images_vi))
        self.g_loss_grad= tf.reduce_mean(self.Image_ir_score * tf.square(gradient(self.fusion_image)-gradient(self.images_ir)))+tf.reduce_mean(self.Image_vi_score*tf.square(gradient(self.fusion_image) - gradient(self.images_vi)))
        self.g_loss_sept=tf.reduce_mean(tf.square(self.sept_ir - self.images_ir))+tf.reduce_mean(tf.square(self.sept_vi - self.images_vi))
        
        self.g_loss_2=self.g_loss_int+80*self.g_loss_grad+1*self.g_loss_sept
        
        tf.summary.scalar('g_loss_int',self.g_loss_int)
        tf.summary.scalar('g_loss_grad',self.g_loss_grad)
        tf.summary.scalar('g_loss_sept',self.g_loss_sept)
        self.g_loss_total=self.g_loss_2
        tf.summary.scalar('loss_g',self.g_loss_total)
    self.saver = tf.train.Saver(max_to_keep=50)

    with tf.name_scope('image'):
        tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
        tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0))  
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))
        tf.summary.image('sept_ir',tf.expand_dims(self.sept_ir[1,:,:,:],0))
        tf.summary.image('sept_vi',tf.expand_dims(self.sept_vi[1,:,:,:],0)) 
        tf.summary.image('Image_vi_score',tf.expand_dims(self.Image_vi_score[1,:,:,:],0))
        tf.summary.image('Image_ir_score',tf.expand_dims(self.Image_ir_score[1,:,:,:],0))  
    
  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess,config,"Train_vi")

    if config.is_train:     
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")

    train_data_ir = read_data(data_dir_ir)
    train_data_vi = read_data(data_dir_vi)

    t_vars = tf.trainable_variables()
    self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    print(self.g_vars)


    with tf.name_scope('train_step'):
        self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)

    self.summary_op = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if config.is_train:
      print("Training...")

      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in xrange(0, batch_idxs):
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          counter += 1
          _, err_g,summary_str= self.sess.run([self.train_fusion_op, self.g_loss_total,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi})
          self.train_writer.add_summary(summary_str,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_g:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g))
            #print(a)

        self.save(config.checkpoint_dir, ep)

  def fusion_model(self,img_ir,img_vi):
####################  Layer1  ###########################
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1_ir'):
            weights=tf.get_variable("w1_ir",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_ir",[16],initializer=tf.constant_initializer(0.0))
            conv1_ir= tf.nn.conv2d(img_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_ir = lrelu(conv1_ir)   
        with tf.variable_scope('layer1_vi'):
            weights=tf.get_variable("w1_vi",[5,5,1,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b1_vi",[16],initializer=tf.constant_initializer(0.0))
            conv1_vi= tf.nn.conv2d(img_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv1_vi = lrelu(conv1_vi)           
            
####################  Layer2  ###########################            
        with tf.variable_scope('layer2_ir'):
            weights=tf.get_variable("w2_ir",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_ir",[16],initializer=tf.constant_initializer(0.0))
            conv2_ir= tf.nn.conv2d(conv1_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_ir = lrelu(conv2_ir)         
            
        with tf.variable_scope('layer2_vi'):
            weights=tf.get_variable("w2_vi",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b2_vi",[16],initializer=tf.constant_initializer(0.0))
            conv2_vi= tf.nn.conv2d(conv1_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv2_vi = lrelu(conv2_vi)            
                                  
####################  Layer3  ###########################               
        conv_12_ir=tf.concat([conv1_ir,conv2_ir],axis=-1)
        conv_12_vi=tf.concat([conv1_vi,conv2_vi],axis=-1)        
            
        with tf.variable_scope('layer3_ir'):
            weights=tf.get_variable("w3_ir",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_ir",[16],initializer=tf.constant_initializer(0.0))
            conv3_ir= tf.nn.conv2d(conv_12_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_ir =lrelu(conv3_ir)
        with tf.variable_scope('layer3_vi'):
            weights=tf.get_variable("w3_vi",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b3_vi",[16],initializer=tf.constant_initializer(0.0))
            conv3_vi= tf.nn.conv2d(conv_12_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv3_vi = lrelu(conv3_vi)
            

####################  Layer4  ########################### 
        conv_123_ir=tf.concat([conv1_ir,conv2_ir,conv3_ir],axis=-1)
        conv_123_vi=tf.concat([conv1_vi,conv2_vi,conv3_vi],axis=-1)                   
            
        with tf.variable_scope('layer4_ir'):
            weights=tf.get_variable("w4_ir",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_ir",[16],initializer=tf.constant_initializer(0.0))
            conv4_ir= tf.nn.conv2d(conv_123_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer4_vi'):
            weights=tf.get_variable("w4_vi",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b4_vi",[16],initializer=tf.constant_initializer(0.0))
            conv4_vi= tf.nn.conv2d(conv_123_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4_vi = lrelu(conv4_vi)
            
 
        conv_ir_vi =tf.concat([conv1_ir,conv1_vi,conv2_ir,conv2_vi,conv3_ir,conv3_vi,conv4_ir,conv4_vi],axis=-1)
 
####################  Layer5  ###########################         
        with tf.variable_scope('layer5_fuse'):
            weights=tf.get_variable("w5_fuse",[1,1,128,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b5_fuse",[1],initializer=tf.constant_initializer(0.0))
            conv5_fuse= tf.nn.conv2d(conv_ir_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv5_fuse=tf.nn.tanh(conv5_fuse)
            
####################  Layer6  ########################### 
        with tf.variable_scope('layer6_sept'):
            weights=tf.get_variable("w6_sept",[1,1,1,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b6_sept",[128],initializer=tf.constant_initializer(0.0))
            conv6_sept= tf.nn.conv2d(conv5_fuse, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv6_sept=lrelu(conv6_sept)
            
####################  Layer7  ###########################            
        with tf.variable_scope('layer7_ir'):
            weights=tf.get_variable("w7_ir",[3,3,128,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b7_ir",[16],initializer=tf.constant_initializer(0.0))
            conv7_ir= tf.nn.conv2d(conv6_sept, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_ir = lrelu(conv7_ir)         
            
        with tf.variable_scope('layer7_vi'):
            weights=tf.get_variable("w7_vi",[3,3,128,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b7_vi",[16],initializer=tf.constant_initializer(0.0))
            conv7_vi= tf.nn.conv2d(conv6_sept, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv7_vi = lrelu(conv7_vi) 

####################  Layer8  ###########################            
        with tf.variable_scope('layer8_ir'):
            weights=tf.get_variable("w8_ir",[3,3,16,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b8_ir",[4],initializer=tf.constant_initializer(0.0))
            conv8_ir= tf.nn.conv2d(conv7_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_ir = lrelu(conv8_ir)         
            
        with tf.variable_scope('layer8_vi'):
            weights=tf.get_variable("w8_vi",[3,3,16,4],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b8_vi",[4],initializer=tf.constant_initializer(0.0))
            conv8_vi= tf.nn.conv2d(conv7_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv8_vi = lrelu(conv8_vi)
             
####################  Layer9  ###########################            
        with tf.variable_scope('layer9_ir'):
            weights=tf.get_variable("w9_ir",[3,3,4,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b9_ir",[1],initializer=tf.constant_initializer(0.0))
            conv9_ir = tf.nn.conv2d(conv8_ir, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9_ir = tf.nn.tanh(conv9_ir)         
            
        with tf.variable_scope('layer9_vi'):
            weights=tf.get_variable("w9_vi",[3,3,4,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias=tf.get_variable("b9_vi",[1],initializer=tf.constant_initializer(0.0))
            conv9_vi= tf.nn.conv2d(conv8_vi, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv9_vi = tf.nn.tanh(conv9_vi)            
            
    return conv5_fuse,conv9_ir,conv9_vi
    

  def save(self, checkpoint_dir, step):
    model_name = "IRVIS.model"
    checkpoint_dir = os.path.join(checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
