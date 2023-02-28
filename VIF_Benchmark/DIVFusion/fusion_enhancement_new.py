# author:xxy,time:2022/3/15
# author:xxy,time:2022/2/22
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
from skimage.color import rgb2ycbcr, ycbcr2rgb
############ 常量的预定义 ############
E = tf.constant(0.6, dtype=tf.float32)
l = tf.constant(1.0, dtype=tf.float32)
batch_size = 10
patch_size_x = 64
patch_size_y = 64

############ Tool ############
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


# def gradient(input_tensor, direction):
#     smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
#     smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = tf.reduce_min(gradient_orig)
#     grad_max = tf.reduce_max(gradient_orig)
#     grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm


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
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    # img_max = np.max(img)
    # img_min = np.min(img)
    # img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
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


# def shuffle_unit(x, groups):
#     with tf.variable_scope('shuffle_unit'):
#         # n, h, w, c = x.get_shape().as_list()
#         x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], groups, 512 // groups])
#         x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
#         x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])
#     return x


def contrast(x):
    with tf.variable_scope('contrast'):
        mean_x = tf.reduce_mean(x, [1, 2], name='global_pool', keep_dims=True)
        c = tf.sqrt(tf.reduce_mean((x - mean_x) ** 2, axis=[1, 2], name='global_pool', keep_dims=True))
    return c


def contrast_1(x):
    with tf.variable_scope('contrast_1'):
        mean_x = tf.reduce_mean

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


############ 获得decomposition的feature ############
reader = tf.train.NewCheckpointReader('./checkpoint/decom_net_train/model.ckpt')


def encoder(img):
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
        # with tf.variable_scope('layer5'):
        #     weights = tf.get_variable("w5", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer5/w5')))
        #     bias = tf.get_variable("b5", initializer=tf.constant(reader.get_tensor('DecomNet/encoder/layer5/b5')))
        #     conv5 = tf.contrib.layers.batch_norm(
        #         tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
        #         updates_collections=None, epsilon=1e-5, scale=True)
        #     conv5 = lrelu(conv5)
            feature = conv4
    return feature


############ Decoder ############
def decoder_ir(feature_ir):
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


def decoder_vi_l(feature_vi_e, feature_l):
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
def CAM_IR(input_feature):
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


def CAM_VI_E(input_feature):
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


def CAM_L(input_feature):
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
def get_fusion_feature(vi,ir):
    # 两个图像都得要是通道为1的
    img = tf.concat([vi,ir],axis=-1)
    feature = encoder(img)
    vector_ir = CAM_IR(feature)
    feature_ir = get_sf_ir(vector_ir, feature)
    vector_vi_e = CAM_VI_E(feature)
    feature_vi_e = get_sf_vi_e(vector_vi_e, feature)
    return feature_ir, feature_vi_e


############ 细节保持模块 ############
def gradient_model(feature_fusion):
    with tf.variable_scope('gradient_model'):
        with tf.variable_scope('layer_laplacian'):
            feature_fusion_laplacian = laplacian(feature_fusion)
        feature_new1 = tf.add(feature_fusion, feature_fusion_laplacian)
        with tf.variable_scope('layer_sobel'):
            feature_fusion_sobel = gradient_feature(feature_fusion)
            with tf.variable_scope('layer_1'):
                weights = tf.get_variable("w1", [1, 1, 256, 128],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
                feature_fusion_sobel_new = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(feature_fusion_sobel, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable("w1", [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_new1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable("w1", [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable("w1", [1, 1, 256, 128],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
        feature_fusion_gradient = tf.concat([conv3, feature_fusion_sobel_new], axis=3)
    return feature_fusion_gradient


############ 对比度增强模块 ############
def contrast_enhancement(feature_fusion_gradient):
    with tf.variable_scope('contrast_model'):
        with tf.variable_scope('layer_1'):
            weights = tf.get_variable("w1", [3, 3, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer_2'):
            weights = tf.get_variable("w1", [5, 5, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer_3'):
            weights = tf.get_variable("w1", [7, 7, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer_4'):
            weights = tf.get_variable("w1", [1, 1, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_gradient, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        feature_multiscale = tf.concat([conv1, conv2, conv3, conv4], axis=3)
        # feature_shuffle = shuffle_unit(x=feature_multiscale, groups=4)
        feature_shuffle = feature_multiscale

        with tf.variable_scope('layer_contrast'):
            mean_vector = tf.reduce_mean(feature_shuffle, [1, 2], name='global_pool', keep_dims=True)
            # feature_contrast = tf.sqrt(tf.reduce_sum((feature_shuffle - mean_vector)**2, axis=[1, 2])) / tf.float32(feature_shuffle.shape[1]*feature_shuffle.shape[2])
            feature_contrast = tf.sqrt(tf.reduce_mean((feature_shuffle - mean_vector) ** 2, [1, 2], name='global_pool', keep_dims=True))
            contrast_vector = tf.reduce_mean(feature_contrast, [1, 2], name='global_pool', keep_dims=True)
        feature_fusion_enhancement = tf.multiply(contrast_vector, feature_shuffle)
    return feature_fusion_enhancement


############ 大Decoder ############
def decoder(feature_fusion_enhancement):
    with tf.variable_scope('decoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion_enhancement, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 128, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 32, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            Y_f = tf.sigmoid(conv4)
        return Y_f


############ 变量的预定义 ############
sess = tf.InteractiveSession()
vi = tf.placeholder(tf.float32, [None, None, None, 1], name='vi')
ir = tf.placeholder(tf.float32, [None, None, None, 1], name='ir')
vi_3 = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_3')
patch_yf = tf.placeholder(tf.float32, [None, None, None, 1], name='patch_yf')
patch_if = tf.placeholder(tf.float32, [None, None, None, 3], name='patch_if')
If = tf.placeholder(tf.float32, [None, None, None, 3], name='If')
If_input = tf.placeholder(tf.float32, [None, None, None, 3], name='If_input')
vi_cb = tf.placeholder(tf.float32, [None, None, None, 1], name='vi_cb')
vi_cr = tf.placeholder(tf.float32, [None, None, None, 1], name='vi_cr')


def enhance(vi, ir):
    with tf.variable_scope('enhance'):
        [feature_ir, feature_vi_e] = get_fusion_feature(vi, ir)
        feature_fusion = tf.concat((feature_ir, feature_vi_e), axis=-1)
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [1, 1, 512, 256],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_fusion, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            feature_fusion = lrelu(conv)
        feature_fusion_gradient = gradient_model(feature_fusion)
        feature_fusion_enhancement = contrast_enhancement(feature_fusion_gradient)
        Y_f= decoder(feature_fusion_enhancement)
    return Y_f


# def color_stay(If_input):
#     with tf.variable_scope('color'):
#         with tf.variable_scope('layer1'):
#             weights = tf.get_variable("w1", [3, 3, 3, 32],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
#             conv1 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(If_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv31 = lrelu(conv1)
#         with tf.variable_scope('layer2'):
#             weights = tf.get_variable("w2", [3, 3, 32, 64],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
#             conv2 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv2 = lrelu(conv2)
#         with tf.variable_scope('layer3'):
#             weights = tf.get_variable("w3", [3, 3, 64, 3],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b3", [3], initializer=tf.constant_initializer(0.0))
#             conv3 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             If = tf.sigmoid(conv3)
#     return If

Y_f = enhance(vi, ir)
If = get_if_tensor(Yf=Y_f, vi_3=vi_3)
# If_input = get_if_tensor(Yf=Y_f, vi_3=vi_3)
# If = color_stay(If_input)

############ Loss ############
gradient_loss = tf.reduce_mean(tf.square(gradient(Y_f)-tf.maximum(gradient(ir),gradient(vi))))
exposure_loss = tf.reduce_mean(tf.abs(Y_f-E))
l1 = tf.reduce_mean(tf.abs(Y_f-ir))
contrast_loss = tf.reduce_mean(tf.abs(contrast(Y_f)-tf.maximum(contrast(vi), contrast(ir))))
color_angle_loss = tf.reduce_mean(angle(If[:,:,:,0], vi_3[:,:,:,0]) + angle(If[:,:,:,1], vi_3[:,:,:,1]) + angle(If[:,:,:,2], vi_3[:,:,:,2]))
l1_loss = tf.reduce_mean(angle(Y_f, ir))
color_mutual_loss = tf.reduce_mean(tf.abs(1./(contrast(If)+0.0001)))
loss_enhance = 200*gradient_loss + 0*1*contrast_loss + 1.1*color_angle_loss + 2.5*l1_loss

############ 训练预备 ############
lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_enhance = [var for var in tf.trainable_variables() if 'enhance' in var.name]
train_op_enhance = optimizer.minimize(loss_enhance, var_list=var_enhance)
sess.run(tf.global_variables_initializer())
saver_enhance = tf.train.Saver(var_list=var_enhance)
print("[*] Initialize model successfully...")


############ 准备数据 ############
# load_data
train_ir_data = []
train_vi_data = []
train_vi_3_data = []
# train_vi_data_names = glob('D:\\Pycharm\\dataset\\train_vi\\*.jpg') #  专供测试代码时候用
# train_ir_data_names = glob('D:\\Pycharm\\dataset\\train_ir\\*.jpg') #  专供测试代码时候用
train_ir_data_names = glob('./ours_dataset_240/train/infrared/*.jpg') #  实际训练使用
train_vi_data_names = glob('./ours_dataset_240/train/visible/*.jpg')  #  实际训练使用
train_ir_data_names.sort()
train_vi_data_names.sort()
print('[*] Number of training data_ir/vi: %d' % len(train_ir_data_names))
for idx in range(len(train_ir_data_names)):
    im_before_ir = load_images(train_ir_data_names[idx])
    ir_gray = cv2.cvtColor(im_before_ir, cv2.COLOR_RGB2GRAY)
    train_ir_data.append(ir_gray)
    im_before_vi = load_images(train_vi_data_names[idx])
    train_vi_3_data.append(im_before_vi)
    vi_gray = cv2.cvtColor(im_before_vi, cv2.COLOR_RGB2GRAY)
    vi_y = rgb_ycbcr_np_3(im_before_vi)[:,:,0]
    train_vi_data.append(vi_y)  # 是归一化之后的图像形成一个list组

# eval_data
eval_ir_data = []
eval_vi_data = []
eval_vi_3_data = []
# eval_ir_data_name = glob('D:\\Pycharm\\dataset\\eval_ir\\*.jpg')
# eval_vi_data_name = glob('D:\\Pycharm\\dataset\\eval_vi\\*.jpg')

# eval_ir_data_name = glob('./ours_dataset_240/test/infrared/*.jpg')
# eval_vi_data_name = glob('./ours_dataset_240/test/visible/*.jpg')
###################
# 学长的100张图像测试
# eval_ir_data_name = glob('./ours_dataset_240/test_100/infrared/*.png')
# eval_vi_data_name = glob('./ours_dataset_240/test_100/visible/*.png')
###################
###################
# 自己的50张图像测试
eval_ir_data_name = glob('./ours_dataset_240/test_50/infrared/*.jpg')
eval_vi_data_name = glob('./ours_dataset_240/test_50/visible/*.jpg')
###################
###################
# TNO测试
# eval_ir_data_name = glob('./ours_dataset_240/test_tno/infrared/*.jpg')
# eval_vi_data_name = glob('./ours_dataset_240/test_tno/visible/*.jpg')
###################
###################
# Roadscene测试
# eval_ir_data_name = glob('./ours_dataset_240/test_roadscene/infrared/*.jpg')
# eval_vi_data_name = glob('./ours_dataset_240/test_roadscene/visible/*.jpg')
###################
data_ir_dir = './ours_dataset_240/test_50/infrared'
data_vi_dir = './ours_dataset_240/test_50/visible'
# data_ir_dir = './ours_dataset_240/test_3/infrared'
# data_vi_dir = './ours_dataset_240/test_3/visible'
eval_ir_data_name.sort(key=lambda x:int(x[len(data_ir_dir)+1:-4]))
eval_vi_data_name.sort(key=lambda x:int(x[len(data_vi_dir)+1:-4]))
for idx in range(len(eval_ir_data_name)):
    eval_im_before_ir = load_images(eval_ir_data_name[idx])
    eval_ir_gray = cv2.cvtColor(eval_im_before_ir, cv2.COLOR_RGB2GRAY)
    eval_ir_data.append(eval_ir_gray)
    eval_im_before_vi = load_images(eval_vi_data_name[idx])
    eval_vi_3_data.append(eval_im_before_vi)
    eval_vi_gray = cv2.cvtColor(eval_im_before_vi, cv2.COLOR_RGB2GRAY)
    eval_vi_y = rgb_ycbcr_np_3(eval_im_before_vi)[:,:,0]
    eval_vi_data.append(eval_vi_y)



epoch = 200
learning_rate = 0.0001
sample_dir = './enhance_train_if/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
sample_dir_yf = './enhance_train_yf/'
if not os.path.isdir(sample_dir_yf):
    os.makedirs(sample_dir_yf)
eval_every_epoch = 50
train_phase = 'enhance'
numBatch = len(train_ir_data) // int(batch_size)  # 批数据量是10,一个小patch图片大小是48
train_op = train_op_enhance
train_loss = loss_enhance
saver = saver_enhance

checkpoint_dir = './checkpoint/enhance_train/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


############ 训练过程 ############
start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
# epoch是2000
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):  # 总共的图片数目除以一个批数据10，所得的批数
        batch_input_ir = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype="float32")
        batch_input_vi = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype="float32")
        batch_input_vi_3 = np.zeros((batch_size, patch_size_y, patch_size_x, 3), dtype="float32")
        # batch_input_yf = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype="float32")
        batch_input_If = np.zeros((batch_size, patch_size_y, patch_size_x, 3), dtype="float32")
        for patch_id in range(batch_size):
            train_ir_data[image_id] = np.reshape(train_ir_data[image_id], [1024, 1280, 1])
            train_vi_data[image_id] = np.reshape(train_vi_data[image_id], [1024, 1280, 1])
            h, w, _= train_ir_data[image_id].shape
            y = random.randint(0, h - patch_size_y - 1)
            #  返回参数1与参数2之间的任一整数，我草这也是个不错的处理方式，比浩哥的那个虽然不那么合理，但是简单
            x = random.randint(0, w - patch_size_x - 1)
            batch_input_ir[patch_id, :, :, :] = train_ir_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            batch_input_vi[patch_id, :, :, :] = train_vi_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            batch_input_vi_3[patch_id, :, :, :] = train_vi_3_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            # batch_input_If[patch_id, :, :, :]
            image_id = (image_id + 1) % len(train_ir_data)
        batch_input_yf = sess.run(Y_f, feed_dict={vi: batch_input_vi, ir: batch_input_ir})
        batch_input_If = get_if(Yf=batch_input_yf, vi_3=batch_input_vi_3)
        _, loss, loss_gradient, loss_exposure, loss_contrast, loss_color_mutual, loss_color_angle, loss_l1 = sess.run(
            [train_op, train_loss, gradient_loss, exposure_loss, contrast_loss, color_mutual_loss, color_angle_loss, l1_loss],
            feed_dict={vi: batch_input_vi,\
                       ir: batch_input_ir,\
                       vi_3: batch_input_vi_3,\
                       If: batch_input_If,\
                       patch_if: batch_input_If, \
                       patch_yf: batch_input_yf, \
                       lr: learning_rate})
        # input_low_hist: input_per_hist1
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        print("gradient:%.4f, exposure:%.4f, contrast:%.4f, mutual:%.4f, angle:%.4f, l1:%.4f" \
              % (loss_gradient, loss_exposure, loss_contrast, loss_color_mutual, loss_color_angle, loss_l1))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        # 训练了一段时间之后看当时epoch的结果
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        for idx in range(len(eval_vi_data)):
            # input_ir_eval = np.reshape(eval_ir_data[idx], [1, 1024, 1280, 1])
            # input_vi_eval = np.reshape(eval_vi_data[idx], [1, 1024, 1280, 1])
            # input_vi_eval = np.expand_dims(eval_vi_data[idx], axis=0)
            # input_ir_eval = np.expand_dims(eval_ir_data[idx], axis=0)
            input_vi_eval = np.expand_dims(eval_vi_data[idx], axis=[0,-1])
            input_ir_eval = np.expand_dims(eval_ir_data[idx], axis=[0,-1])
            input_vi_3_eval = np.expand_dims(eval_vi_3_data[idx], axis=0)
            result_1 = sess.run(Y_f, feed_dict={vi: input_vi_eval, ir: input_ir_eval})
            result = sess.run(If, feed_dict={vi_3: input_vi_3_eval, Y_f: result_1})
            # save_images(os.path.join(sample_dir_yf, 'yf_%d_%d.png' % (idx + 1, epoch + 1)), result_1)
            save_images(os.path.join(sample_dir_yf, '%d.jpg' % (idx + 1)), result_1)
            # save_images(os.path.join(sample_dir, 'if_%d_%d.png' % (idx + 1, epoch + 1)), result)
            save_images(os.path.join(sample_dir, '%d.jpg' % (idx + 1)), result)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
print("[*] Finish training for phase %s." % train_phase)









