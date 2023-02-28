import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.05

class m_Encoder(object):

    def __init__(self, sco):
        self.scope = sco
        self.weight_vars = []
        # with tf.variable_scope(self.scope):
        with tf.variable_scope('m_encoder'):
            self.weight_vars.append(self._create_variables(1, 32, 3, scope='conv1_1'))
            self.weight_vars.append(self._create_variables(32, 128, 1, scope='conv2_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(32, 128, 1, scope='conv3_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='conv3_2'))
            self.weight_vars.append(self._create_variables(32, 128, 1, scope='conv4_1'))
            self.weight_vars.append(self._create_variables(128, 32, 3, scope='conv4_2'))
    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV),
                                 name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def encode(self, image):
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == 0:
                out1 = conv2d(out, kernel, bias, dense=False, use_relu=True,
                             Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==1:
                out = conv2d_trans(out1, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==2:
                out2 = conv2d(out, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==3:
                out = conv2d_trans(out2, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==4:
                out3 = conv2d(out, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==5:
                out = conv2d_trans(out3, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
            elif i==6:
                out4 = conv2d(out, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/m_encoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)
        return out1,out2,out3,out4


def conv2d(x, kernel, bias, dense=False, use_relu=True, Scope=None, BN=True, Reuse=None):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    # conv and add bias
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
    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=False, reuse=Reuse)
    if use_relu:
        out = tf.nn.relu(out)
    return out
