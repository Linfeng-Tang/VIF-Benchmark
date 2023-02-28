import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

WEIGHT_INIT_STDDEV = 0.1


class m_Decoder(object):

    def __init__(self, sco):
        self.weight_vars = []
        self.scope = sco
        with tf.variable_scope('m_decoder'):

            self.weight_vars.append(self._create_variables(128, 64, 3, scope='conv2_1'))
            self.weight_vars.append(self._create_variables(64, 32, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(32, 16, 3, scope='conv2_3'))
            self.weight_vars.append(self._create_variables(16, 1, 3, scope='conv2_4'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name='bias')
        return (kernel, bias)

    def decode(self, image):
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False,
                             Scope = self.scope + '/m_decoder/b' + str(i), BN=False, Reuse=tf.AUTO_REUSE)
            else:
                out = conv2d(out, kernel, bias, use_relu=True,
                             Scope = self.scope + '/m_decoder/b' + str(i), BN=True, Reuse=tf.AUTO_REUSE)

        out = tf.nn.tanh(out) / 2 + 0.5
        return out


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
