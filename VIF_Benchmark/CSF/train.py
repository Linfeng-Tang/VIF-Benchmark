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

LEARNING_RATE = 0.0003
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8


def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
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
		SOURCE = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE')
		Enco = Encoder('Encoder')
		feas = Enco.encode(image = SOURCE, is_training = True)

		Deco = Decoder('Decoder')
		RECON_SOURCE = Deco.decode(features = feas, is_training=True)

		LOSS = 25 * tf.reduce_mean(tf.square(SOURCE - RECON_SOURCE)) + tf.reduce_mean(1-SSIM_LOSS(SOURCE, RECON_SOURCE))
		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Encoder')
		theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Decoder')
		theta = theta_e + theta_d
		for v in theta:
			print(v.name)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			solver = tf.train.AdamOptimizer(learning_rate).minimize(LOSS, global_step = current_iter, var_list = theta)

		clip = [p.assign(tf.clip_by_value(p, -50, 50)) for p in theta]

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep = 20)

		# tf.summary.scalar('vis_loss', vis_loss)
		# tf.summary.scalar('ir_loss', ir_loss)
		tf.summary.scalar('loss', LOSS)
		tf.summary.scalar('Learning rate', learning_rate)
		tf.summary.image('source', tf.expand_dims(SOURCE[:, :, :, 0], axis=-1), max_outputs = 3)
		tf.summary.image('reconstructed', tf.expand_dims(RECON_SOURCE[:, :, :, 0], axis=-1), max_outputs = 3)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

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
				VIS_batch = np.expand_dims(VIS_batch, -1)
				IR_batch = np.expand_dims(IR_batch, -1)
				VIS_batch = (VIS_batch - 0.5) * 2
				IR_batch = (IR_batch - 0.5) * 2


				feed_batch = np.zeros_like(VIS_batch)
				rand = np.random.randint(2, size = BATCH_SIZE)
				for bn in range(BATCH_SIZE):
					feed_batch[bn, :, :, :] = VIS_batch[bn, :, :, :] * rand[bn] + IR_batch[bn, :, :, :] * (1-rand[bn])

				FEED_DICT = {SOURCE: feed_batch}

				# run the training step
				sess.run([solver, clip], feed_dict = FEED_DICT)
				Loss = sess.run(LOSS, feed_dict = FEED_DICT)
				print("[epoch:%s, batch:%s] loss:%s" % (epoch+1, batch, Loss))

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)
				is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
				if is_last_step or step % logging_period == 0:
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

		writer.close()
		saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')

