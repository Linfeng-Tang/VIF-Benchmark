# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from networks import Encoder, Classification, Decoder
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

samples = 10

def generate(ir_path, vis_path, model_path_EC, model_path_ED, output_path = None):
	ir_img = (imread(ir_path, flatten=True, mode='YCbCr') / 255.0 - 0.5) * 2
	vis_img = (imread(vis_path, flatten=True, mode='YCbCr') / 255.0 - 0.5) * 2
	H, W = vis_img.shape

	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	ir_dimension.insert(0, 1)
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)

	with tf.Graph().as_default() as graph:
		with tf.Session() as sess:
			SOURCE = tf.placeholder(tf.float32, shape = (None, H, W, 1), name = 'SOURCE')
			FEAS = tf.placeholder(tf.float32, shape = (None, H, W, 24), name = 'FEATURES')

			Enco = Encoder('Encoder')
			Deco = Decoder('Decoder')
			feas = Enco.encode(image = SOURCE, is_training = False)
			RESULT = Deco.decode(features = FEAS, is_training = False)

			Class = Classification('Classification')
			out = Class.classification(image = FEAS, is_training = False)
			prob = tf.nn.softmax(out)  # argmax(prob, 1)

			sess.run(tf.global_variables_initializer())

			# restore the trained model
			theta_e = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Encoder')
			theta_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Decoder')
			savere = tf.train.Saver(var_list = theta_e+theta_d)
			ED_model_num = str(4)
			savere.restore(sess, model_path_ED)

			theta_c = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Classification')
			saverc = tf.train.Saver(var_list = theta_c)
			saverc.restore(sess, model_path_EC)

			FEED_DICT_VIS = {SOURCE: vis_img}
			FEED_DICT_IR = {SOURCE: ir_img}
			vis_feas_output = sess.run(feas, feed_dict = FEED_DICT_VIS)
			ir_feas_output = sess.run(feas, feed_dict = FEED_DICT_IR)

			'''mean strategy'''
			# print("mean strategy")
			# fuse_feas = 0.5 * vis_feas_output + 0.5 * ir_feas_output

			'''addition strategy'''
			# print("addition strategy")
			# fuse_feas = vis_feas_output + ir_feas_output

			'''max strategy'''
			# print("max strategy")
			# Difference = vis_feas_output - ir_feas_output
			# ders_fuse_vis = np.int8(Difference > 0)
			# ders_fuse_ir = np.int8(Difference < 0)
			# fuse_feas = np.maximum(vis_feas_output, ir_feas_output)

			'''l1_norm strategy'''
			# print("l1_norm strategy")
			# fuse_feas, ders_fuse_vis, ders_fuse_ir = L1_norm(vis_feas_output, ir_feas_output)

			'''CS-based strategy'''
			var_vis = np.zeros([samples, H, W, 1])
			var_ir = np.zeros([samples, H, W, 1])
			ders_vis = np.zeros([samples, H, W, 24])
			ders_ir = np.zeros([samples, H, W, 24])
			diff_vis_ir = ir_img - (vis_img- np.mean(vis_img) + np.mean(ir_img))
			diff_ir_vis = vis_img - np.mean(vis_img) + np.mean(ir_img) - ir_img

			var_vis_pro = np.zeros(shape=(1, samples), dtype=np.float32)
			for i in range(samples):
				var_ir[i,:,:,:] = vis_img - np.mean(vis_img) + np.mean(ir_img) + diff_vis_ir * (i + 1) / samples
				var_vis[i,:,:,:] = ir_img + (i + 1) / samples * diff_ir_vis

			ders_list_ir = sess.run(tf.gradients(out[0, 1], FEAS), feed_dict = {FEAS: sess.run(feas, feed_dict = {SOURCE:var_ir})})
			ders_list_vis = sess.run(tf.gradients(out[0, 0], FEAS), feed_dict={FEAS: sess.run(feas, feed_dict={SOURCE: var_vis})})

			ders_list_ir=ders_list_ir[0]
			ders_list_vis=ders_list_vis[0]
			for i in range(samples):
				ders_ir[i, :, :, :] = np.abs(ders_list_ir[i]) * (0.25 ** i)
				ders_vis[i, :, :, :] = np.abs(ders_list_vis[i]) * (0.25 ** i)

			ders_vis = np.expand_dims(np.mean(ders_vis, axis=0), axis=0)
			ders_ir = np.expand_dims(np.mean(ders_ir, axis = 0), axis=0)

			mean_vis = np.mean(ders_vis)
			mean_ir = np.mean(ders_ir)
			ders_ir = ders_ir + mean_vis - mean_ir

			c = 0.00015
			ders_fuse_vis, ders_fuse_ir = softmax(ders_vis / c, ders_ir / c)
			fuse_feas = ders_fuse_vis * vis_feas_output + ders_fuse_ir * ir_feas_output

			result = sess.run(RESULT, feed_dict={FEAS: fuse_feas})
			imsave(output_path, result[0, :, :, 0] / 2 + 0.5)



def scale(A):
	A = (A - np.min(A)) / (np.max(A) - np.min(A))
	return A

def scale2(A, B):
	b, h, w, c = A.shape
	M = np.max([np.max(A), np.max(B)])
	N = np.min([np.min(A), np.min(B)])
	A = (A - N) / (M - N)
	B = (B - N) / (M - N)
	return A, B


def softmax(x, y):
	x_exp = np.exp(x)
	y_exp = np.exp(y)
	x_out = x_exp / (x_exp + y_exp)
	y_out = y_exp / (x_exp + y_exp)
	return (x_out, y_out)

def count():
	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	return total_parameters


def L1_norm(source_en_a, source_en_b):
	narry_a = source_en_a
	narry_b = source_en_b
	dimension = source_en_a.shape

	# caculate L1-norm
	temp_abs_a = tf.abs(narry_a)
	temp_abs_b = tf.abs(narry_b)
	_l1_a = tf.reduce_sum(temp_abs_a, 3)
	_l1_b = tf.reduce_sum(temp_abs_b, 3)

	_l1_a = tf.reduce_sum(_l1_a, 0)
	_l1_b = tf.reduce_sum(_l1_b, 0)
	l1_a = _l1_a.eval()
	l1_b = _l1_b.eval()
	# caculate the map for source images
	mask_value = l1_a + l1_b
	mask_sign_a = l1_a / mask_value
	mask_sign_b = l1_b / mask_value
	array_MASK_a = mask_sign_a
	array_MASK_b = mask_sign_b
	print(array_MASK_a.shape)

	for i in range(dimension[3]):
		temp_matrix = array_MASK_a * narry_a[0, :, :, i] + array_MASK_b * narry_b[0, :, :, i]
		temp_matrix = temp_matrix.reshape([1, dimension[1], dimension[2], 1])
		if i == 0:
			result = temp_matrix
		else:
			result = np.concatenate([result, temp_matrix], axis = -1)
	return result, array_MASK_a.reshape([1, dimension[1], dimension[2], 1]), array_MASK_b.reshape([1, dimension[1], dimension[2], 1])



def save_images(paths, datas, save_path, prefix = None, suffix = None):
	if isinstance(paths, str):
		paths = [paths]

	assert (len(paths) == len(datas))

	if not exists(save_path):
		mkdir(save_path)

	if prefix is None:
		prefix = ''
	if suffix is None:
		suffix = ''

	for i, path in enumerate(paths):
		data = datas[i]
		# print('data ==>>\n', data)
		if data.shape[2] == 1:
			data = data.reshape([data.shape[0], data.shape[1]])
		# print('data reshape==>>\n', data)

		name, ext = splitext(path)
		name = name.split(sep)[-1]

		path = join(save_path, prefix + suffix + ext)
		print('data path==>>', path)
		imsave(path, data)


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g