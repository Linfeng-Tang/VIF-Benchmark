import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.05


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		self.kernel_size = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(1, 16, 5, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(16, 24, 5, scope = 'conv1_4'))
				self.weight_vars.append(self._create_variables(24, 24, 5, scope = 'conv1_6'))
				self.weight_vars.append(self._create_variables(24, 24, 5, scope = 'conv1_8'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.kernel_size.append(kernel_size)
		return (kernel, bias)

	def encode(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, self.kernel_size[i], BN =False, Scope = self.scope + '/encoder/b' + str(i), training = is_training)
		return out



class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.kernel_size = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(24, 16, 5, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(16, 16, 5, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(16, 16, 5, scope = 'conv1_3'))
				self.weight_vars.append(self._create_variables(16, 1, 5, scope = 'conv1_4'))


	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.kernel_size.append(kernel_size)
		return (kernel, bias)

	def decode(self, features, is_training):
		final_layer_idx = len(self.weight_vars) - 1
		out = features
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, self.kernel_size[i], use_lrelu = False, BN=False, Scope = self.scope + '/decoder/b' + str(i), training = is_training)
				out = tf.nn.tanh(out)#  / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, self.kernel_size[i], BN = False, Scope = self.scope + '/decoder/b' + str(i), training = is_training)
		return out




class Classification(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		self.kernel_size = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('classification'):
				self.weight_vars.append(self._create_variables(24, 32, 5, scope = 'conv1_6'))
				self.weight_vars.append(self._create_variables(32, 64, 5, scope = 'conv1_8'))
				self.weight_vars.append(self._create_variables(64, 32, 5, scope = 'conv1_9'))
				self.weight_vars.append(self._create_variables(32, 2, 5, scope = 'conv1_11'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.kernel_size.append(kernel_size)
		return (kernel, bias)

	def classification(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, self.kernel_size[i], Scope = self.scope + '/classification/b' + str(i),
			             training = is_training)
			if i in [0, 1, 2, 3]:
				out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'+str(i))
			else:
				out = out
			if i == len(self.weight_vars)-1:
				out = tf.reduce_mean(out, axis = [1, 2])
		return out







def conv2d(x, kernel, bias, kernel_size, use_lrelu = True, Scope = None, BN = True, training = True, strides = [1, 1, 1, 1]):
	# padding image with reflection mode
	# for i in range(kernel_size//2):
	x_padded = tf.pad(x, [[0, 0], [kernel_size//2, kernel_size//2], [kernel_size//2, kernel_size//2], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = training)
	if use_lrelu:
		out = tf.maximum(out, 0.2*out)
	# out = tf.nn.relu(out)
	return out