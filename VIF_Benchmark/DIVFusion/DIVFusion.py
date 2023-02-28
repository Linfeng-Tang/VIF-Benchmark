# author:xxy,time:2022/3/26
############ tf的预定义 ############
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
from glob import glob
import cv2
import losses
from model import *
from numpy import *
from natsort import natsorted
from skimage.color import rgb2ycbcr, ycbcr2rgb
import argparse
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

############ 常量的预定义 ############
E = tf.constant(0.6, dtype=tf.float32)
l = tf.constant(1.0, dtype=tf.float32)
batch_size = 10
patch_size_x = 64
patch_size_y = 64

############ Tool ############
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def gradient(input_tensor):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    gradient_orig_x = tf.abs(tf.nn.conv2d(input_tensor, smooth_kernel_x, strides=[1, 1, 1, 1], padding='SAME'))
    gradient_orig_y = tf.abs(tf.nn.conv2d(input_tensor, smooth_kernel_y, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min_x = tf.reduce_min(gradient_orig_x)
    grad_max_x = tf.reduce_max(gradient_orig_x)
    grad_min_y = tf.reduce_min(gradient_orig_y)
    grad_max_y = tf.reduce_max(gradient_orig_y)
    grad_norm_x = tf.div((gradient_orig_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))
    grad_norm_y = tf.div((gradient_orig_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
    grad_norm = grad_norm_x + grad_norm_y
    return grad_norm


def gradient_feature(input_tensor):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_x = tf.broadcast_to(smooth_kernel_x, [2, 2, 256, 256])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    gradient_orig_x = tf.abs(tf.nn.conv2d(input_tensor, smooth_kernel_x, strides=[1, 1, 1, 1], padding='SAME'))
    gradient_orig_y = tf.abs(tf.nn.conv2d(input_tensor, smooth_kernel_y, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min_x = tf.reduce_min(gradient_orig_x)
    grad_max_x = tf.reduce_max(gradient_orig_x)
    grad_min_y = tf.reduce_min(gradient_orig_y)
    grad_max_y = tf.reduce_max(gradient_orig_y)
    grad_norm_x = tf.div((gradient_orig_x - grad_min_x), (grad_max_x - grad_min_x + 0.0001))
    grad_norm_y = tf.div((gradient_orig_y - grad_min_y), (grad_max_y - grad_min_y + 0.0001))
    grad_norm = grad_norm_x + grad_norm_y
    return grad_norm


def laplacian(input_tensor):
    # kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32)
    kernel = tf.reshape(tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]), [3, 3, 1, 1])
    kernel = tf.broadcast_to(kernel, [3, 3, 256, 256])
    gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def load_images(file):
    im = Image.open(file).convert('RGB')
    img = np.array(im, dtype="float32") / 255.0
    img_norm = np.float32(img)
    return img_norm


def hist(input):
    input_int = np.uint8((input*255.0))
    input_hist = cv2.equalizeHist(input_int)
    input_hist = (input_hist/255.0).astype(np.float32)
    return input_hist


def save_images(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


def contrast(x):
    with tf.variable_scope('contrast'):
        mean_x = tf.reduce_mean(x, [1, 2], name='global_pool', keep_dims=True)
        c = tf.sqrt(tf.reduce_mean((x - mean_x) ** 2, axis=[1, 2], name='global_pool', keep_dims=True))
    return c


def angle(a,b):
    vector = tf.multiply(a,b)
    up = tf.reduce_sum(vector)
    down = tf.sqrt(tf.reduce_sum(tf.square(a))) * tf.sqrt(tf.reduce_sum(tf.square(b)))
    theta = tf.acos(up/down)  # 弧度制
    return theta


def rgb_ycbcr(img_rgb):
    R = tf.expand_dims(img_rgb[:, :, :, 0], axis=-1)
    G = tf.expand_dims(img_rgb[:, :, :, 1], axis=-1)
    B = tf.expand_dims(img_rgb[:, :, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
    return img_ycbcr


def ycbcr_rgb(img_ycbcr):
    Y = tf.expand_dims(img_ycbcr[:, :, :, 0], axis=-1)
    Cb = tf.expand_dims(img_ycbcr[:, :, :, 1], axis=-1)
    Cr = tf.expand_dims(img_ycbcr[:, :, :, 2], axis=-1)
    R = Y + 1.402*(Cr - 128/255)
    G = Y - 0.34414*(Cb - 128/255) - 0.71414*(Cr - 128/255)
    B = Y + 1.772*(Cb - 128/255)
    img_rgb = tf.concat([R,G,B], axis=-1)
    return img_rgb


def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


def rgb_ycbcr_np_3(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


def ycbcr_rgb_np(img_ycbcr):
    Y = np.expand_dims(img_ycbcr[:, :, :, 0], axis=-1)
    Cb = np.expand_dims(img_ycbcr[:, :, :, 1], axis=-1)
    Cr = np.expand_dims(img_ycbcr[:, :, :, 2], axis=-1)
    R = Y + 1.402*(Cr - 128/255)
    G = Y - 0.34414*(Cb - 128/255) - 0.71414*(Cr - 128/255)
    B = Y + 1.772*(Cb - 128/255)
    img_rgb = np.concatenate([R, G, B], axis=-1)
    return img_rgb


def get_if(Yf, vi_3):
    vi_ycbcr = rgb_ycbcr_np(vi_3)
    cb = np.expand_dims(vi_ycbcr[:, :, :, 1], axis=-1)
    cr = np.expand_dims(vi_ycbcr[:, :, :, 2], axis=-1)
    If = np.concatenate([Yf, cb, cr], axis=-1)
    If = ycbcr_rgb_np(If)
    return If


def get_if_tensor(Yf, vi_3):
    vi_ycbcr = rgb_ycbcr(vi_3)
    cb = tf.expand_dims(vi_ycbcr[:, :, :, 1], axis=-1)
    cr = tf.expand_dims(vi_ycbcr[:, :, :, 2], axis=-1)
    If = tf.concat([Yf, cb, cr], axis=-1)
    If = ycbcr_rgb(If)
    return If


def encoder(img, reader):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1",  initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer2/b2')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer3/b3')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer4/w4')))
            bias = tf.get_variable("b4", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer4/b4')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
            feature = conv4
    return feature


############ Decoder ############
def decoder_ir(feature_ir, reader):
    with tf.variable_scope('decoder_ir'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_ir, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer2/b2')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer3/b3')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer4/w4')))
            bias = tf.get_variable("b4", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_ir/layer4/b4')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            ir_r = tf.nn.tanh(conv4)
        return ir_r


def decoder_vi_l(feature_vi_e, feature_l, reader):
    with tf.variable_scope('decoder_vi_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_vi_e, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer2/b2')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer3/b3')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer4/w4')))
            bias = tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('DecomNet/decoder_vi_l/layer4/b4')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vi_e_r = tf.sigmoid(conv4)
    with tf.variable_scope('decoder_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer1/w1')))
            l_conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_l, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv1 = lrelu(l_conv1)
            l_conv1 = tf.concat([l_conv1, conv1], axis=3)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer2/b2')))
            l_conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv2 = lrelu(l_conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer3/b3')))
            l_conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv3 = lrelu(l_conv3)
            l_conv3 = tf.concat([l_conv3, conv3],axis=3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer4/w4')))
            bias = tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('DecomNet/decoder_l/layer4/b4')))
            l_conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_r = tf.sigmoid(l_conv4)
        return vi_e_r, l_r


############ CAM #############
def CAM_IR(input_feature, reader):
    with tf.variable_scope('CAM_IR'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_IR/layer/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_IR/layer/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_IR/layer/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_IR/layer/b2')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vector_ir = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_ir = tf.nn.softmax(vector_ir)
    return vector_ir


def CAM_VI_E(input_feature, reader):
    with tf.variable_scope('CAM_VI_E'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_VI_E/layer/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_VI_E/layer/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_VI_E/layer/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_VI_E/layer/b2')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vector_vi_e = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_vi_e = tf.nn.softmax(vector_vi_e)
    return vector_vi_e


def CAM_L(input_feature, reader):
    with tf.variable_scope('CAM_L'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_L/layer/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_L/layer/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_L/layer/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader.get_tensor('DecomNet/CAM_L/layer/b2')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vector_l = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_l = tf.nn.softmax(vector_l)
    return vector_l


############ Special Feature ############
def get_sf_ir(vector_ir, feature):
    with tf.variable_scope('special_feature_ir'):
        # new_vector_ir = tf.broadcast_to(vector_ir, feature.shape)
        feature_ir = tf.multiply(vector_ir, feature)
    return feature_ir

def get_sf_l(vector_l, feature):
    with tf.variable_scope('special_feature_l'):
        # new_vector_l = tf.broadcast_to(vector_l, feature.shape)
        feature_l = tf.multiply(vector_l, feature)
    return feature_l

def get_sf_vi_e(vector_vi_e, feature):
    with tf.variable_scope('special_feature_vi_e'):
        # new_vector_vi_e = tf.broadcast_to(vector_vi_e, feature.shape)
        feature_vi_e = tf.multiply(vector_vi_e, feature)
    return feature_vi_e


############ feature_get_model ############
def get_fusion_feature(vi,ir, reader):
    # 两个图像都得要是通道为1的
    img = tf.concat([vi,ir],axis=-1)
    feature = encoder(img, reader)
    vector_ir = CAM_IR(feature, reader)
    feature_ir = get_sf_ir(vector_ir, feature)
    vector_vi_e = CAM_VI_E(feature, reader)
    feature_vi_e = get_sf_vi_e(vector_vi_e, feature)
    return feature_ir, feature_vi_e


############ 细节保持模块 ############
def gradient_model(feature_fusion, reader_enhance):
    with tf.variable_scope('gradient_model'):
        with tf.variable_scope('layer_laplacian'):
            feature_fusion_laplacian = laplacian(feature_fusion)
        feature_new1 = tf.add(feature_fusion, feature_fusion_laplacian)
        with tf.variable_scope('layer_sobel'):
            feature_fusion_sobel = gradient_feature(feature_fusion)
            with tf.variable_scope('layer_1'):
                weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_sobel/layer_1/w1')))
                bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_sobel/layer_1/b1')))
                feature_fusion_sobel_new = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(feature_fusion_sobel, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_new1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_2/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_2/b1')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable("w1",initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_3/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/gradient_model/layer_3/b1')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
        feature_fusion_gradient = tf.concat([conv3, feature_fusion_sobel_new], axis=3)
    return feature_fusion_gradient


############ 对比度增强模块 ############
def contrast_enhancement(feature_fusion_gradient, reader_enhance):
    with tf.variable_scope('contrast_model'):
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_2/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_2/b1')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_3/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_3/b1')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer_4'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_4/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/contrast_model/layer_4/b1')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        feature_multiscale = tf.concat([conv1, conv2, conv3, conv4], axis=3)
        # feature_shuffle = shuffle_unit(x=feature_multiscale, groups=4)
        feature_shuffle = feature_multiscale

        with tf.variable_scope('layer_contrast'):
            mean_vector = tf.reduce_mean(feature_shuffle, [1, 2], name='global_pool', keep_dims=True)
            feature_contrast = tf.sqrt(tf.reduce_mean((feature_shuffle - mean_vector) ** 2, [1, 2], name='global_pool', keep_dims=True))
            contrast_vector = tf.reduce_mean(feature_contrast, [1, 2], name='global_pool', keep_dims=True)
        feature_fusion_enhancement = tf.multiply(contrast_vector, feature_shuffle)
    return feature_fusion_enhancement


############ 大Decoder ############
def decoder(feature_fusion_enhancement, reader_enhance):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer1/w1')))
            bias = tf.get_variable("b1", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer1/b1')))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_enhancement, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer2/w2')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer2/b2')))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer3/w3')))
            bias = tf.get_variable("b3", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer3/b3')))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer4/w4')))
            bias = tf.get_variable("b4", initializer=tf.constant(reader_enhance.get_tensor('enhance/decoder/layer4/b4')))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            Y_f = tf.sigmoid(conv4)
        return Y_f



def enhance(vi, ir, reader, reader_enhance):
    with tf.variable_scope('enhance'):
        [feature_ir, feature_vi_e] = get_fusion_feature(vi, ir, reader)
        feature_fusion = tf.concat((feature_ir, feature_vi_e), axis=-1)
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", initializer=tf.constant(reader_enhance.get_tensor('enhance/layer/w1')))
            bias = tf.get_variable("b2", initializer=tf.constant(reader_enhance.get_tensor('enhance/layer/b2')))
            conv = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            feature_fusion = lrelu(conv)
        feature_fusion_gradient = gradient_model(feature_fusion, reader_enhance)
        feature_fusion_enhancement = contrast_enhancement(feature_fusion_gradient, reader_enhance)
        Y_f= decoder(feature_fusion_enhancement, reader_enhance)
    return Y_f





###################
# 自己的50张图像测试
def prepare_data(ir_dir, vi_dir):
    # eval_data
    eval_ir_data = []
    eval_vi_data = []
    eval_vi_3_data = []
    filelist = natsorted(os.listdir(ir_dir))
    
    for item in filelist:
        ir_name = os.path.join(ir_dir, item)
        vi_name = os.path.join(vi_dir, item)
        eval_im_before_ir = load_images(ir_name)
        eval_ir_gray = cv2.cvtColor(eval_im_before_ir, cv2.COLOR_RGB2GRAY)
        eval_ir_data.append(eval_ir_gray)
        eval_im_before_vi = load_images(vi_name)
        eval_vi_3_data.append(eval_im_before_vi)
        eval_vi_gray = cv2.cvtColor(eval_im_before_vi, cv2.COLOR_RGB2GRAY)
        eval_vi_y = rgb_ycbcr_np_3(eval_im_before_vi)[:,:,0]
        eval_vi_data.append(eval_vi_y)
    return eval_ir_data, eval_vi_data, eval_vi_3_data, filelist


def main(Method = 'SeAFusion', model_path_1='', model_path_2='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    eval_ir_data, eval_vi_data, eval_vi_3_data, eval_ir_data_name = prepare_data(ir_dir, vi_dir)
    ############ 获得decomposition的feature ############
    reader = tf.train.NewCheckpointReader(model_path_1)
    reader_enhance = tf.train.NewCheckpointReader(model_path_2)
    
    ############ 变量的预定义 ############
    sess = tf.InteractiveSession()
    vi = tf.placeholder(tf.float32, [None, None, None, 1], name='vi')
    ir = tf.placeholder(tf.float32, [None, None, None, 1], name='ir')
    vi_3 = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_3')
    If = tf.placeholder(tf.float32, [None, None, None, 3], name='If')
    
    Y_f = enhance(vi, ir, reader, reader_enhance)
    If = get_if_tensor(Yf=Y_f, vi_3=vi_3)
    sess.run(tf.global_variables_initializer())
    test_bar = tqdm(eval_vi_data)
    for idx, data in enumerate(test_bar):
        # img_name = eval_ir_data_name[idx].split('/')[-1]
        input_vi_eval = np.expand_dims(eval_vi_data[idx], axis=[0,-1])
        input_ir_eval = np.expand_dims(eval_ir_data[idx], axis=[0,-1])
        input_vi_3_eval = np.expand_dims(eval_vi_3_data[idx], axis=0)
        
        start = time.time()
        result_1 = sess.run(Y_f, feed_dict={vi: input_vi_eval, ir: input_ir_eval})
        result = sess.run(If, feed_dict={vi_3: input_vi_3_eval, Y_f: result_1})
        end = time.time()
        name = os.path.basename(eval_ir_data_name[idx])
        save_path = os.path.join(save_dir, name)
        save_images(save_path, result)
        test_bar.set_description('{} | {} | {:.4f} s'.format(Method, name, end-start))
            

            
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








