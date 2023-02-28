# author:xxy,time:2022/2/23
import numpy as np
from PIL import Image
import tensorflow as tf
import scipy.stats as st
from skimage import io,data,color
from functools import reduce
import cv2

############ 常量的预定义 ############
batch_size = 5
patch_size_x = 224
patch_size_y = 224
############ Encoder ############
# 输入img为concat红外可见光图像的结果，通道数为2
# 输出为256个feature—map
def encoder(img):
    with tf.variable_scope('encoder'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 2, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(img, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [256], initializer=tf.constant_initializer(0.0))
            conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [256], initializer=tf.constant_initializer(0.0))
            conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            feature = lrelu(conv4)
        # with tf.variable_scope('layer5'):
        #     weights = tf.get_variable("w5", [3, 3, 512, 512],
        #                               initializer=tf.truncated_normal_initializer(stddev=1e-3))
        #     bias = tf.get_variable("b5", [512], initializer=tf.constant_initializer(0.0))
        #     conv5 = tf.contrib.layers.batch_norm(
        #         tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
        #         updates_collections=None, epsilon=1e-5, scale=True)
        #     conv5 = lrelu(conv5)
        #     feature = conv5
    return feature


############ Decoder ############
def decoder_ir(feature_ir):
    with tf.variable_scope('decoder_ir'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_ir, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
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
            ir_r = tf.sigmoid(conv4)
        return ir_r

# def decoder_l(feature_l):
#     with tf.variable_scope('decoder_l'):
#         with tf.variable_scope('layer1'):
#             weights = tf.get_variable("w1", [3, 3, 512, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
#             conv1 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(feature_l, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv1 = lrelu(conv1)
#         with tf.variable_scope('layer2'):
#             weights = tf.get_variable("w2", [3, 3, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
#             conv2 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv2 = lrelu(conv2)
#         with tf.variable_scope('layer3'):
#             weights = tf.get_variable("w3", [3, 3, 128, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b3", [64], initializer=tf.constant_initializer(0.0))
#             conv3 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv3 = lrelu(conv3)
#         with tf.variable_scope('layer4'):
#             weights = tf.get_variable("w4", [3, 3, 64, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
#             conv4 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             l_r = tf.sigmoid(conv4)
#         return l_r
#
# def decoder_vi_e(feature_vi_e):
#     with tf.variable_scope('decoder_vi_e'):
#         with tf.variable_scope('layer1'):
#             weights = tf.get_variable("w1", [3, 3, 512, 256],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b1", [256], initializer=tf.constant_initializer(0.0))
#             conv1 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(feature_vi_e, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv1 = lrelu(conv1)
#         with tf.variable_scope('layer2'):
#             weights = tf.get_variable("w2", [3, 3, 256, 128],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
#             conv2 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv2 = lrelu(conv2)
#         with tf.variable_scope('layer3'):
#             weights = tf.get_variable("w3", [3, 3, 128, 64],
#                                       initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b3", [64], initializer=tf.constant_initializer(0.0))
#             conv3 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             conv3 = lrelu(conv3)
#         with tf.variable_scope('layer4'):
#             weights = tf.get_variable("w4", [3, 3, 64, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
#             bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
#             conv4 = tf.contrib.layers.batch_norm(
#                 tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
#                 updates_collections=None, epsilon=1e-5, scale=True)
#             vi_e_r = tf.sigmoid(conv4)
#         return vi_e_r

def decoder_vi_l(feature_vi_e, feature_l):
    with tf.variable_scope('decoder_vi_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128],
                                      initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_vi_e, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
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
            vi_e_r = tf.sigmoid(conv4)
    with tf.variable_scope('decoder_l'):
        with tf.variable_scope('layer1'):
            weights = tf.get_variable("w1", [3, 3, 256, 128], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            l_conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(feature_l, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv1 = lrelu(l_conv1)
            l_conv1 = tf.concat([l_conv1, conv1], axis=3)
        with tf.variable_scope('layer2'):
            weights = tf.get_variable("w2", [3, 3, 256, 64], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0.0))
            l_conv2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv2 = lrelu(l_conv2)
        with tf.variable_scope('layer3'):
            weights = tf.get_variable("w3", [3, 3, 64, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b3", [32], initializer=tf.constant_initializer(0.0))
            l_conv3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_conv3 = lrelu(l_conv3)
            l_conv3 = tf.concat([l_conv3, conv3],axis=3)
        with tf.variable_scope('layer4'):
            weights = tf.get_variable("w4", [3, 3, 64, 1], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b4", [1], initializer=tf.constant_initializer(0.0))
            l_conv4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l_conv3, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            l_r = tf.sigmoid(l_conv4)
        return vi_e_r, l_r


############ CAM #############
def CAM_IR(input_feature):
    with tf.variable_scope('CAM_IR'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vector_ir = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_ir = tf.nn.softmax(vector_ir)
    return vector_ir


def CAM_VI_E(input_feature):
    with tf.variable_scope('CAM_VI_E'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            vector_vi_e = tf.reduce_mean(conv1, [1, 2], name='global_pool', keep_dims=True)
            vector_vi_e = tf.nn.softmax(vector_vi_e)
    return vector_vi_e


def CAM_L(input_feature):
    with tf.variable_scope('CAM_L'):
        with tf.variable_scope('layer'):
            weights = tf.get_variable("w1", [3, 3, 256, 32], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(input_feature, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
            weights = tf.get_variable("w2", [3, 3, 32, 256], initializer=tf.truncated_normal_initializer(stddev=1e-3))
            bias = tf.get_variable("b2", [256], initializer=tf.constant_initializer(0.0))
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


############ All_model ############
def decomposition(vi,ir):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        # 两个图像都得要是通道为1的
        img = tf.concat([vi,ir],axis=-1)
        feature = encoder(img)
        vector_ir = CAM_IR(feature)
        feature_ir = get_sf_ir(vector_ir, feature)
        ir_r = decoder_ir(feature_ir)

        vector_vi_e = CAM_VI_E(feature)
        feature_vi_e = get_sf_vi_e(vector_vi_e, feature)
        vector_l = CAM_L(feature)
        feature_l = get_sf_l(vector_l, feature)
        [vi_e_r, l_r] = decoder_vi_l(feature_vi_e, feature_l)

        # vector_l = CAM_L(feature)
        # feature_l = get_sf_l(vector_l, feature)
        # l_r = decoder_l(feature_l)
    return ir_r, vi_e_r, l_r


############ Tool ############
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def gradient(input_tensor, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    gradient_orig = tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
    grad_min = tf.reduce_min(gradient_orig)
    grad_max = tf.reduce_max(gradient_orig)
    grad_norm = tf.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def laplacian(input_tensor):
    kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32)
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


def rgb_ycbcr(img_rgb):
    R = tf.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = tf.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = tf.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    img_ycbcr = tf.concat([Y, Cb, Cr], axis=-1)
    return img_ycbcr


def rgb_ycbcr_np(img_rgb):
    R = np.expand_dims(img_rgb[:, :, 0], axis=-1)
    G = np.expand_dims(img_rgb[:, :, 1], axis=-1)
    B = np.expand_dims(img_rgb[:, :, 2], axis=-1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255.0
    img_ycbcr = np.concatenate([Y, Cb, Cr], axis=-1)
    return img_ycbcr


# def shuffle_unit(x, groups):
#     with tf.variable_scope('shuffle_unit'):
#         n, h, w, c = x.get_shape().as_list()
#         x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, groups, c // groups]))
#         x = tf.transpose(x, tf.convert_to_tensor([0, 1, 2, 4, 3]))
#         x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], h, w, c]))
#     return x