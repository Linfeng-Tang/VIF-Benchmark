import tensorflow as tf
import numpy as np



def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)


def L1_LOSS(batchimg):
	# input(B H W)
	L1_norm = tf.reduce_sum(tf.abs(batchimg), axis = [1, 2])
	# tf.norm(batchimg, axis = [1, 2], ord = 1) / int(batchimg.shape[1])
	# (B 1)
	E = tf.reduce_mean(L1_norm)
	# (1)
	return E


# Fro_loss
def Fro_LOSS(batchimg):
	# (B H W)
	fro_norm = tf.square(tf.norm(batchimg, axis = [1, 2], ord = 'fro'))
	E = tf.reduce_mean(fro_norm)
	return E
