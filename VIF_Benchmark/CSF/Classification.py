import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.2


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				self.weight_vars.append(self._create_variables(1, 32, 3, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(32, 64, 3, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(64, 128, 3, scope = 'conv1_3'))
				# self.weight_vars.append(self._create_variables(64, 2, 3, scope = 'conv1_4'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def encode(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, Scope = self.scope + '/encoder/b' + str(i), training=is_training)
			# if i in [1, 2, 3]:
			# 	out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'+str(i))
			print("Encoder: after " + str(i + 1) + "-th layer: ", out.shape)
			# if i == 3:
				# out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
				# with tf.variable_scope(self.scope):
				# 	with tf.variable_scope('flatten1'):
				# 		out = tf.layers.dense(out, 2, activation = None, use_bias = True, trainable = True,
				# 		                      reuse = reuse)
			# 		# with tf.variable_scope('flatten2'):
			# 		# 	out = tf.layers.dense(out, 256, activation = tf.nn.relu, use_bias = True, trainable = True,
			# 		# 	                      reuse = reuse)
			# if i == 5:
			# 	out = tf.reduce_mean(out, axis = [1, 2])
			# 	print("final shape: ", out.shape)
		return out


class Classification(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.weight_vars = []
		with tf.variable_scope(self.scope):
			with tf.variable_scope('classification'):
				self.weight_vars.append(self._create_variables(128, 32, 7, scope = 'conv1_1'))
				self.weight_vars.append(self._create_variables(32, 32, 7, scope = 'conv1_2'))
				self.weight_vars.append(self._create_variables(32, 16, 7, scope = 'conv1_3'))
				self.weight_vars.append(self._create_variables(16, 8, 7, scope = 'conv1_4'))
				self.weight_vars.append(self._create_variables(2, 2, 7, scope = 'conv1_5'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
			                     name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def classification(self, image, is_training):
		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			out = conv2d(out, kernel, bias, BN=False, Scope = self.scope + '/classification/b' + str(i), training = is_training)
			out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'+str(i))
			print("Classification: after " + str(i + 1) + "-th layer: ", out.shape)
			#if i == 2:
		# out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		# with tf.variable_scope(self.scope):
		# 	with tf.variable_scope('flatten1'):
		# 		out = tf.layers.dense(out, 2, activation = None, use_bias = True, trainable = True,
		# 		                      reuse = reuse)
		# 		# with tf.variable_scope('flatten2'):
		# 		# 	out = tf.layers.dense(out, 256, activation = tf.nn.relu, use_bias = True, trainable = True,
		# 		# 	                      reuse = reuse)
			if i == 2:
				out = tf.reduce_mean(out, axis = [1, 2])
				print("Classification: final shape: ", out.shape)
		return out





# class Decoder(object):
# 	def __init__(self, scope_name):
# 		self.weight_vars = []
# 		self.scope = scope_name
# 		with tf.name_scope(scope_name):
# 			with tf.variable_scope('decoder'):
# 				self.weight_vars.append(self._create_variables(240, 240, 3, scope = 'conv2_1'))
# 				self.weight_vars.append(self._create_variables(240+192, 128, 3, scope = 'conv2_2'))
# 				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
# 				self.weight_vars.append(self._create_variables(64+96, 32, 3, scope = 'conv2_4'))
# 				self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_5'))
#
#
# 	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
# 		with tf.variable_scope(scope):
# 			shape = [kernel_size, kernel_size, input_filters, output_filters]
# 			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
# 			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
# 		return (kernel, bias)
#
# 	def decode(self, code2,code4,code5):
# 		final_layer_idx = len(self.weight_vars) - 1
#
# 		out0 = code5
# 		for i in range(len(self.weight_vars)):
# 			kernel, bias = self.weight_vars[i]
# 			if i == 0:
# 				out1 = conv2d(out0, kernel, bias, use_relu = True, Scope = self.scope + '/decoder/b' + str(i), BN = False)
# 				out1=up_sample(out1, scale_factor = 2)
# 			if i == 1:
# 				out2 = conv2d(tf.concat([out1, code4],3), kernel, bias, use_relu = True, BN = True, Scope = self.scope + '/decoder/b' + str(i))
# 			if i == 2:
# 				out3 = conv2d(out2, kernel, bias, use_relu = True, BN = True, Scope = self.scope + '/decoder/b' + str(i))
# 				out3 = up_sample(out3, scale_factor = 2)
# 			if i == 3:
# 				out4 = conv2d(tf.concat([out3,code2],3), kernel, bias, use_relu = True, BN = True, Scope = self.scope + '/decoder/b' + str(i))
# 			if i == final_layer_idx:
# 				out = conv2d(out4, kernel, bias, use_relu = False, Scope = self.scope + '/decoder/b' + str(i), BN = False)
# 				out = tf.nn.tanh(out) / 2 + 0.5
#
# 		return out


def conv2d(x, kernel, bias, use_lrelu = True, Scope = None, BN = True, training = True, strides = [1, 1, 1, 1]):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
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


# def up_sample(x, scale_factor = 2):
# 	_, h, w, _ = x.get_shape().as_list()
# 	new_size = [h * scale_factor, w * scale_factor]
# 	return tf.image.resize_nearest_neighbor(x, size = new_size)