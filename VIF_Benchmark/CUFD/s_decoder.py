import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.05

class s_Decoder(object):

    def __init__(self, sco):
        self.weight_vars = []
        self.scope = sco
        with tf.variable_scope('s_decoder'):
            self.weight_vars.append(self._create_variables(64, 32, 5, scope='conv1_1'))

            self.weight_vars.append(self._create_variables(32, 128, 1, scope='dense_block1_conv1_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block1_conv1_2'))
            self.weight_vars.append(self._create_variables(32, 128, 1, scope='dense_block1_conv2_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block1_conv2_2'))
            self.weight_vars.append(self._create_variables(64, 128, 1, scope='dense_block1_conv3_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block1_conv3_2'))

            self.weight_vars.append(self._create_variables(96, 32, 1, scope='transfer_layer1'))

            self.weight_vars.append(self._create_variables(32, 128, 1, scope='dense_block2_conv1_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block2_conv1_2'))
            self.weight_vars.append(self._create_variables(32, 128, 1, scope='dense_block2_conv2_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block2_conv2_2'))
            self.weight_vars.append(self._create_variables(64, 128, 1, scope='dense_block2_conv3_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='dense_block2_conv3_2'))

            self.weight_vars.append(self._create_variables(96, 32, 1, scope='output_layer'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def decode(self, img1,img2):
        img = tf.concat([img1,img2], 3)
        final_layer_idx = len(self.weight_vars) - 1
        out = img
        dense_indices = [2, 4, 6, 9, 11, 13]
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == 0:
                out = conv2d(out, kernel, bias, dense=False, use_relu=True,
                                   Scope=self.scope + '/s_decoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i in dense_indices:
                if i == 2 or i == 9:
                    out = conv2d_2(out, kernel_last, kernel, bias_last, bias,
                                   dense=False, use_relu=True,
                                   Scope=self.scope + '/s_decoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
                elif i in dense_indices:
                    out = conv2d_2(out, kernel_last, kernel, bias_last, bias,
                                   dense=True, use_relu=True,
                                   Scope=self.scope + '/s_decoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i == 7:
                out = conv2d_trans(out, kernel, bias, dense=False, use_relu=True,
                                   Scope=self.scope + '/s_decoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i == 14:
                out = conv2d_trans(out, kernel, bias, dense=False, use_relu=False,
                                   Scope=self.scope + '/s_decoder/b' + str(i), BN=False, Reuse=tf.AUTO_REUSE)
            kernel_last = kernel[:]
            bias_last = bias[:]
        out = tf.nn.tanh(out) / 2 + 0.5
        return out


def conv2d(x, kernel, bias, dense=False, use_relu=True, Scope=None, BN=True, Reuse=None):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=False, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out

def conv2d_trans(x, kernel, bias, dense=False, use_relu=True, Scope=None, BN=True, Reuse=None):
    # padding image with reflection mode
    x_padded = x
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=False, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out


def conv2d_2(x, kernel1, kernel2, bias1, bias2, dense=False, use_relu=True, Scope=None, BN=True, Reuse=None):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias1)
    out = tf.nn.conv2d(out, kernel2, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias2)

    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=False, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    if dense:
        out = tf.concat([out, x], 3)
    return out