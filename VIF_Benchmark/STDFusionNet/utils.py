"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS


def read_data(path):
    """
    Read h5 format data file
  
    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        return data


def preprocess(path, scale=3):
    """
    Preprocess single image file 
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
  
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)
    # Must be normalized
    image = image / 255
    label_ = image / 255
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


def prepare_data(sess, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset
  
      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    print("data length: ", len(data))

    return data


def make_data(sess, data, data_dir):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        # savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    else:
        savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'test.h5'))
        if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
            os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def input_setup(sess, config, data_dir, index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    if config.is_train:
        data = prepare_data(sess, dataset=data_dir)
    else:
        data = prepare_data(sess, dataset=data_dir)

    sub_input_sequence = []

    if config.is_train:
        for i in range(len(data)):
            input_ = (imread(data[i]) - 127.5) / 127.5
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                    # Make channel value
                    if data_dir == "Train":
                        sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                                               interpolation=cv2.INTER_CUBIC)
                        sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                        print('error')
                    else:
                        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])

                    sub_input_sequence.append(sub_input)

    else:
        input_ = (imread(data[index]) - 127.5) / 127.5
        if len(input_.shape) == 3:
            h_real, w_real, _ = input_.shape
        else:
            h_real, w_real = input_.shape
        input_ = np.lib.pad(input_, ((padding, padding_h), (padding, padding_w)), 'edge')
        h, w = input_.shape
        # print(input_.shape)
        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_input_sequence.append(sub_input)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    # print(arrdata.shape)
    make_data(sess, arrdata, data_dir)

    if not config.is_train:
        print(nx, ny)
        print(h_real, w_real)
        return nx, ny, h_real, w_real


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return (img * 127.5 + 127.5)


def gradient(input):
    filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
    filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
    Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
    Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
    Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
    return Gradient


def weights_spectral_norm(weights, u=None, iteration=1, update_collection=None, reuse=False, name='weights_SN'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        if u is None:
            u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                                trainable=False)

        def power_iteration(u, ite):
            v_ = tf.matmul(u, tf.transpose(w_mat))
            v_hat = l2_norm(v_)
            u_ = tf.matmul(v_hat, w_mat)
            u_hat = l2_norm(u_)
            return u_hat, v_hat, ite + 1

        u_hat, v_hat, _ = power_iteration(u, iteration)

        sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))

        w_mat = w_mat / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            if not (update_collection == 'NO_OPS'):
                print(update_collection)
                tf.add_to_collection(update_collection, u.assign(u_hat))

            w_norm = tf.reshape(w_mat, w_shape)
        return w_norm


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def sigmoid_relu(x):
    return tf.where(tf.greater(tf.nn.sigmoid(x), tf.nn.relu(x)), tf.nn.sigmoid(x), tf.nn.relu(x))

def l2_norm(input_x, epsilon=1e-12):
    input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
    return input_x_norm
