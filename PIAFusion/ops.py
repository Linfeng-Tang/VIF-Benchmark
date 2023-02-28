import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()

weight_init = tf.truncated_normal_initializer(stddev=1e-3)
weight_regularizer = None


##################################################################################
# Layer
##################################################################################

def conv(x, channels=1, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv', sn=False, norm=False):
	with tf.variable_scope(scope):
		if pad > 0:
			if (kernel - stride) % 2 == 0:
				pad_top = pad
				pad_bottom = pad
				pad_left = pad
				pad_right = pad

			else:
				pad_top = pad
				pad_bottom = kernel - stride - pad_top
				pad_left = pad
				pad_right = kernel - stride - pad_left

			if pad_type == 'zero':
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
			if pad_type == 'reflect':
				x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
		w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
							regularizer=weight_regularizer)
		if sn:
			w = weights_spectral_norm(w)
		x = tf.nn.conv2d(input=x, filter=w, strides=[1, stride, stride, 1], padding='VALID')
		if use_bias:
			bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
			x = tf.nn.bias_add(x, bias)
		if norm:
			x = batch_norm(x)

		return x


def up_sample(x, up_x=None, stride=2, padding='SAME'):
	x_shape = x.get_shape().as_list()
	if padding == 'SAME':
		# print('x shape:', x_shape[1])
		if x_shape[1] is None:
			output_shape = tf.shape(up_x[0, :, :, 0])
		else:
			output_shape = [x_shape[1] * stride, x_shape[2] * stride]
	else:
		output_shape = [x_shape[1] * stride + max(kernel - stride, 0),
						x_shape[2] * stride + max(kernel - stride, 0)]
	up_sample = tf.image.resize_images(x, output_shape, method=1)
	return up_sample


def depthwise_conv(x, kernel=3, stride=1, scope='depthwise_conv', sn=True, norm=False):
	with tf.variable_scope(scope):
		w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], 1], initializer=weight_init,
							regularizer=weight_regularizer)
		if sn:
			w = weights_spectral_norm(w)
		# bias = tf.get_variable("bias", [x.get_shape().as_list()[-1]], initializer=tf.constant_initializer(0.0))
		x = tf.nn.depthwise_conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
		# x = tf.nn.bias_add(x, bias)
		if norm:
			x = batch_norm(x)
		return x


def deconv(x, up_features=None, channels=1, kernel=4, stride=2, pad=1, use_bias=True, scope='deconv', sn=False, norm=False):
	with tf.variable_scope(scope):
		output_shape = tf.shape(up_features[0, :, :, 0])
		up_sample = tf.image.resize_images(x, output_shape, method=1)
		x = conv(up_sample, channels, kernel=kernel, stride=stride, pad=pad, pad_type='reflect', use_bias=use_bias, sn=sn)
		x = x + up_features
	return x


def attribute_connet(x, channels, use_bias=True, sn=True, scope='attribute'):
	with tf.variable_scope(scope):
		x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init,
							kernel_regularizer=weight_regularizer, use_bias=use_bias)
		return x


def fully_conneted(x, channels, use_bias=True, sn=True, scope='fully'):
	with tf.variable_scope(scope):
		x = tf.layers.flatten(x)
		shape = x.get_shape().as_list()
		x_channel = shape[-1]
		if sn:
			w = tf.get_variable("kernel", [x_channel, channels], tf.float32, initializer=weight_init,
								regularizer=weight_regularizer)
			if use_bias:
				bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

				x = tf.matmul(x, spectral_norm(w)) + bias
			else:
				x = tf.matmul(x, spectral_norm(w))

		else:
			x = tf.layers.dense(x, units=channels, kernel_initializer=weight_init,
								kernel_regularizer=weight_regularizer, use_bias=use_bias)
			print('fully_connected shape: ', x.get_shape().as_list())
		return x

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.1):
	# pytorch alpha is 0.01
	return tf.nn.leaky_relu(x, alpha)


def relu(x):
	return tf.nn.relu(x)


def tanh(x):
	return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
	return tf_contrib.layers.instance_norm(x,
										   epsilon=1e-05,
										   center=True, scale=True,
										   scope=scope)


def batch_norm(x, scope='batch_norm'):
	return tf_contrib.layers.batch_norm(x,
										decay=0.999,
										center=True,
										scale=False,
										epsilon=0.001,
										scope=scope)


def layer_norm(x, scope='layer_norm'):
	return tf_contrib.layers.layer_norm(x,
										center=True, scale=True,
										scope=scope)


def spectral_norm(w, iteration=1):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u
	v_hat = None
	for i in range(iteration):
		"""
		power iteration
		Usually iteration = 1 will be enough
		"""
		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = tf.nn.l2_normalize(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = tf.nn.l2_normalize(u_)

	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = w / sigma
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm







##################################################################################
# Loss function
##################################################################################
def L1_loss(x, y):
	loss = tf.reduce_mean(tf.abs(y - x))
	return loss


def Fro_LOSS(batchimg):
	fro_norm = tf.square(tf.norm(batchimg, axis=[1, 2], ord='fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
	# print('fro_norm shape:', fro_norm.get_shape().as_list())
	E = tf.reduce_mean(fro_norm)
	return E


def gradient(input):
	filter1 = tf.reshape(tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]), [3, 3, 1, 1])
	filter2 = tf.reshape(tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]), [3, 3, 1, 1])
	Gradient1 = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='SAME')
	Gradient2 = tf.nn.conv2d(input, filter2, strides=[1, 1, 1, 1], padding='SAME')
	Gradient = tf.abs(Gradient1) + tf.abs(Gradient2)
	return Gradient


def Gradient_loss(image_A, image_B):
	gradient_A = gradient(image_A)
	gradient_B = gradient(image_B)
	grad_loss = tf.reduce_mean(L1_loss(gradient_A, gradient_B))
	return grad_loss


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


def l2_norm(input_x, epsilon=1e-12):
	input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
	return input_x_norm

