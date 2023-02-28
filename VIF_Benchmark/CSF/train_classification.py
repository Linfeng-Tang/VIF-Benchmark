# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage

from networks import Encoder, Classification, Decoder
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
# from generate import generate

patch_size = 128
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.002
EPSILON = 1e-5
DECAY_RATE = 0.85
eps = 1e-8


def train_classification(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	MODEL_SAVE_PATH = save_path + 'temporary.ckpt'
	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE1')
		# SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE2')
		LABEL = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 2), name = 'LABEL')
		print('source shape:', SOURCE1.shape)

		# upsampling vis and ir images
		Enco = Encoder('Encoder')
		Class = Classification('Classification')
		feas1 = Enco.encode(image = SOURCE1, is_training = False)
		# feas2 = Enco.encode(image = SOURCE2, is_training = False)
		# feas = tf.concat([feas1, feas2], axis = -1)
		prob = Class.classification(image = feas1, is_training = True)
		out = tf.argmax(prob, 1, name = 'predict')

		corre = tf.equal(tf.argmax(prob, 1), tf.argmax(LABEL, 1))
		acc = tf.reduce_mean(tf.cast(corre, tf.float32))
		# ones = tf.ones([BATCH_SIZE, 1])
		# zeros = tf.zeros([BATCH_SIZE, 1])
		# vis_label = tf.concat([ones, zeros], axis = -1)
		# ir_label = tf.concat([zeros, ones], axis = -1)

	#######  LOSS FUNCTION
		# loss = tf.reduce_mean(-tf.reduce_sum(LABEL*tf.log(prob), reduction_indices=[1]))
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = LABEL, logits = prob))
		# tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=prob, labels=LABEL))

		# loss = -tf.reduce_mean(LABEL * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)))

		# ir_loss = tf.reduce_mean(-tf.reduce_sum(ir_label*tf.log(ir_prob), reduction_indices=[1]))
		# #tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = ir_prob, labels = ir_label))
		# labels = tf.random_uniform(D2_real.shape, minval = 0.7, maxval = 1.2, dtype = tf.float32)))

		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)


		theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Classification')
		#theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Encoder')
		# theta=theta_c+theta_e
		for v in theta:
			print(v.name)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			solver = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = current_iter, var_list = theta)

		clip = [p.assign(tf.clip_by_value(p, -100, 100)) for p in theta]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 20)

		theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Encoder')
		saver0 = tf.train.Saver(var_list = theta_e)
		saver0.restore(sess, './models_ED/4/4.ckpt')

		tf.summary.scalar('loss', loss)
		tf.summary.scalar('accuracy', acc)
		tf.summary.scalar('Learning rate', learning_rate)
		tf.summary.image('source1', tf.expand_dims(SOURCE1[:, :, :, 0], axis=-1), max_outputs = 2)
		# tf.summary.image('source2', tf.expand_dims(SOURCE2[:, :, :, 0], axis = -1), max_outputs = 2)


		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs_EC/", sess.graph)

		# ** Start Training **
		step = 0
		count_loss = 0
		num_imgs = source_imgs.shape[0]

		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step
				VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
				IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
				# IR_batch=1-IR_batch
				VIS_batch = (VIS_batch - 0.5) * 2
				IR_batch = (IR_batch - 0.5) * 2
				# vis_mean = np.mean(VIS_batch)
				# ir_mean = np.mean(IR_batch)
				# if vis_mean > ir_mean:
				# 	IR_batch = IR_batch + vis_mean - ir_mean
				# else:
				for b in range(BATCH_SIZE):
					VIS_batch[b, :, :] = VIS_batch[b, :, :] + np.mean(IR_batch[b, :, :]) - np.mean(VIS_batch[b, :, :])

				VIS_batch = np.expand_dims(VIS_batch, -1)
				IR_batch = np.expand_dims(IR_batch, -1)
				#ones = np.ones([BATCH_SIZE, 1])
				#zeros = np.ones([BATCH_SIZE, 1])
				#feed_batch = np.concatenate([np.zeros_like(VIS_batch), np.zeros_like(VIS_batch)], axis=-1)
				feed_batch1 = np.zeros_like(VIS_batch)
				# feed_batch2 = np.zeros_like(IR_batch)
				rand = np.random.randint(2, size = BATCH_SIZE)
				rand_reverse = 1 - rand
				for bn in range(BATCH_SIZE):
					feed_batch1[bn, :, :, :] = VIS_batch[bn, :, :, :] * rand[bn] + IR_batch[bn, :, :, :] * (1-rand[bn])
					# feed_batch2[bn, :, :, :] = VIS_batch[bn, :, :, :] * rand_reverse[bn] + IR_batch[bn, :, :, :] * (1 - rand_reverse[bn])
					# feed_batch[bn, :, :, 1] = VIS_batch[bn, :, :, 0] * rand[bn] + IR_batch[bn, :, :, 0] * (1-rand[bn])


				label = np.concatenate(
					[np.expand_dims(rand * 1.0, axis = -1), np.expand_dims(rand_reverse * 1.0, axis = -1)], axis = -1)
				FEED_DICT = {SOURCE1: feed_batch1, LABEL: label}
				# FEED_DICT_IR = {SOURCE: IR_batch, LABEL: ones}

				# run the training step
				sess.run([solver, clip], feed_dict = FEED_DICT)
				Loss = sess.run(loss, feed_dict = FEED_DICT)
				feed_label = sess.run(LABEL[:, 0], feed_dict=FEED_DICT)


				print("[epoch:%s, batch:%s] loss:%s" % (epoch+1, batch, Loss))
				print("label:", feed_label)
				print("out_label:", sess.run(1-out, feed_dict=FEED_DICT))
				# print("sum: ", sess.run(tf.reduce_sum(tf.abs(out-tf.expand_dims(label[:, 0], axis=-1))), feed_dict=FEED_DICT))
				print("acc: ", sess.run(acc, feed_dict=FEED_DICT))
				print('\n')

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
				# gofvis = sess.run(grad_of_vis, feed_dict = {source: source_batch})
				# IF, gofIF = sess.run([generated_img, grad(generated_img)], feed_dict = {source: source_batch})
				# fig = plt.figure()
				# orig = fig.add_subplot(221)
				# gradslove = fig.add_subplot(222)
				# orig.imshow(source_batch[1, :, :, 0], cmap = 'gray')
				# gradslove.imshow(gofvis[1, :, :, 0], cmap = 'gray')
				# oriF = fig.add_subplot(223)
				# gradF = fig.add_subplot(224)
				# oriF.imshow(sess.run(generated_img, feed_dict = {source: source_batch})[1, :, :, 0], cmap = 'gray')
				# print(sess.run(grad(generated_img)[1, :, :, 0], feed_dict = {source: source_batch}))
				# gradF.imshow(sess.run(grad(generated_img)[1, :, :, 0], feed_dict = {source: source_batch}),
				#              cmap = 'gray')
				# plt.show()

				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
					logits = sess.run(prob, feed_dict = FEED_DICT)
					print("logits:", logits[:, 0])

				# 	elapsed_time = datetime.now() - start_time
				# 	loss = sess.run(total, feed_dict = FEED_DICT)
				# 	g_loss, d1_loss = sess.run([G_loss, D1_loss], feed_dict = FEED_DICT)
				# 	d1_fake, d1_real = sess.run([tf.reduce_mean(D1_fake), tf.reduce_mean(D1_real)],
				# 	                            feed_dict = FEED_DICT)
				# 	d2_fake, d2_real = sess.run([tf.reduce_mean(D2_fake), tf.reduce_mean(D2_real)],
				# 	                            feed_dict = FEED_DICT)
				# 	lr = sess.run(learning_rate)
				# 	# loss_vis = sess.run(LOSS_VIS, feed_dict = FEED_DICT)
				# 	# loss_ir = sess.run(LOSS_IR, feed_dict = FEED_DICT)
				# 	g_img = sess.run(generated_img, feed_dict = FEED_DICT)
				#
				# 	print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
				# 		epoch + 1, EPOCHS, step, lr, elapsed_time))
				# 	print('D1_fake:%s, D1_real:%s' % (d1_fake, d1_real))
				# 	print('D2_fake:%s, D2_real:%s' % (d2_fake, d2_real))
				# 	print('G_loss:%s, D1_loss:%s, D2_loss:%s' % (g_loss, d1_loss, d2_loss))
				# 	print('G_loss_gan: %s, G_loss_norm: %s' % (g_loss_GAN, g_loss_norm))
				# 	maxv = np.max(g_img)
				# 	minv = np.min(g_img)
				# 	print('max_value:%s, min_value:%s\n' % (maxv, minv))

				# if epoch % 1 == 0:
				# 	saver.save(sess, MODEL_SAVE_PATH)
				# 	for i in range(9):
				# 		index = i + 1
				# 		ir_path = path + 'IR_ds_us' + str(index) + '.bmp'
				# 		vis_path = path + 'VIS' + str(index) + '.bmp'
				# 		generate(ir_path, vis_path, MODEL_SAVE_PATH, index, output_path = 'Fused images/' + str(epoch) + '/')

		writer.close()
		saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


# 		# current_iter = tf.Variable(0)
# 		# learning_rate = tf.train.exponential_decay(0.03, current_iter, decay_steps = n_batches, decay_rate = 0.03)
# 		# train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step = current_iter)
#

# 	# ** Done Training & Save the model **
# 	saver.save(sess, save_path)

# loss_data = Loss_all[:count_loss]
# scio.savemat('./models/loss/only_ssim.mat', {'loss': loss_data})


def grad(img):
	# [n, r, c, channel] = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), int(img.shape[3])]
	# img_h = tf.slice(img, [0, 0, 1, 0], [n, r, c - 1, channel])
	# img_v = tf.slice(img, [0, 1, 0, 0], [n, r - 1, c, channel])
	# zeros_h = tf.zeros([n, r, 1, channel])
	# zeros_v = tf.zeros([n, 1, c, channel])
	# img_h = tf.concat([img_h, zeros_h], axis = 2)
	# img_v = tf.concat([img_v, zeros_v], axis = 1)
	# g_h = img - img_h
	# g_v = img - img_v
	# g = tf.sqrt(tf.square(g_h) + tf.square(g_v))

	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	# g = tf.nn.tanh(g) / 2 + 0.5
	return g

