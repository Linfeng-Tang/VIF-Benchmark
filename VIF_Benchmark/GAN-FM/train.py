#coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
from LOSS import  L1_LOSS, Fro_LOSS

patch_size = 84
LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8
n=1.5

def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period = 1):
	from datetime import datetime
	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	# create the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE_VIS')
		SOURCE_IR = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 1), name = 'SOURCE_IR')
		print('source_vis shape:', SOURCE_VIS.shape)


		G = Generator('Generator')
		generated_img = G.transform(vis = SOURCE_VIS, ir = SOURCE_IR)
		print('generate_image:', generated_img.shape)

		# discrimator1 for vis
		D1 = Discriminator1('Discriminator1')
		D1_real,D1_real_logit = D1.discrim_patch(SOURCE_VIS, reuse = False)
		D1_fake,D1_fake_logit = D1.discrim_patch(generated_img, reuse = True)

		# discrimator2 for ir
		D2 = Discriminator2('Discriminator2')
		D2_real,D2_real_logit = D2.discrim_patch(SOURCE_IR, reuse = False)
		D2_fake,D2_fake_logit = D2.discrim_patch(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# G_loss
		G_loss_GAN_D1 = -tf.reduce_mean(tf.log(tf.clip_by_value(D1_fake, 1e-8, 1.0)))
		G_loss_GAN_D2 = -tf.reduce_mean(tf.log(tf.clip_by_value(D1_fake, 1e-8, 1.0)))
		G_loss_GAN = G_loss_GAN_D1 +G_loss_GAN_D2

		# three hyper-parameter
		a=0.7 # visin irin  
		b=5 # grad  in
		c=100 # adv  1 10 1000 10000

		grad_of_vis = grad(SOURCE_VIS)
		grad_of_ir = grad(SOURCE_IR)
		grad_of_gen = grad(generated_img)
		joint_grad =tf.maximum(tf.abs(grad_of_vis),tf.abs(grad_of_ir))

		LOSS_in = (a*Fro_LOSS(generated_img- SOURCE_IR)+(1.0-a)*Fro_LOSS(generated_img- SOURCE_VIS))/(patch_size*patch_size)
		LOSS_grad = L1_LOSS(tf.abs(grad_of_gen) - joint_grad) / (patch_size * patch_size)


		G_loss_content = LOSS_grad+b*LOSS_in
		G_loss = G_loss_GAN + c* G_loss_content

		# Loss for Discriminator
		D1_loss_real = -tf.reduce_mean(tf.log(tf.clip_by_value(D1_real, 1e-8, 1.0)))
		D1_loss_fake = -tf.reduce_mean(tf.log(tf.clip_by_value(1. - D1_fake, 1e-8, 1.0)))
		D1_loss = D1_loss_fake + D1_loss_real

		D2_loss_real = -tf.reduce_mean(tf.log(tf.clip_by_value(D2_real, 1e-8, 1.0)))
		D2_loss_fake = -tf.reduce_mean(tf.log(tf.clip_by_value(1. - D2_fake, 1e-8, 1.0)))
		D2_loss = D2_loss_fake + D2_loss_real

		current_iter = tf.Variable(0)
		# setting decayed learning_rate
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator1')
		theta_D2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator2')


		G_GAN_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_GAN, global_step = current_iter,
		                                                                 var_list = theta_G)
		G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter,
		                                                             var_list = theta_G)
		G_Content_solver=tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_content,global_step=current_iter,var_list=theta_G)

		D1_solver = tf.train.AdamOptimizer(learning_rate).minimize(D1_loss, global_step = current_iter,
		                                                                      var_list = theta_D1)
		D2_solver = tf.train.AdamOptimizer(learning_rate).minimize(D2_loss, global_step = current_iter,
		                                                                      var_list = theta_D2)


		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D1 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D1]
		clip_D2 = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D2]
		# clip all peremeter to (-8 8)

		sess.run(tf.global_variables_initializer())

		saver = tf.train.Saver(max_to_keep = 500)

		# tensorboard
		tf.summary.scalar('D1_loss', D1_loss)
		tf.summary.scalar('D2_loss', D2_loss)
		tf.summary.scalar('loss_grad',LOSS_grad)
		tf.summary.scalar('loss_in',LOSS_in)
		tf.summary.scalar('G_loss_content',G_loss_content)
		tf.summary.scalar('G_loss',G_loss)
		tf.summary.scalar('Learning rate', learning_rate)
		tf.summary.image('vis', SOURCE_VIS, max_outputs = 3)
		tf.summary.image('ir', SOURCE_IR, max_outputs = 3)
		tf.summary.image('grad_of_vis',grad_of_vis,max_outputs=2)
		tf.summary.image('grad_of_gen',grad_of_gen,max_outputs=2)
		tf.summary.image('fused_img', generated_img, max_outputs = 3)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		step = 0

		for epoch in range(EPOCHS):
			it_g = 0
			it_d1 = 0
			it_d2 = 0
			np.random.shuffle(source_imgs)
			for batch in range(n_batches):
				step += 1
				current_iter = step

				VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
				IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
				VIS_batch = np.expand_dims(VIS_batch, -1)
				IR_batch = np.expand_dims(IR_batch, -1)

				FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}


				# run the training step
				sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
				it_d1 += 1
				d1_loss = sess.run(D1_loss, feed_dict=FEED_DICT)
				# while d1_loss > 1.38*n:
				# 	sess.run([D1_solver, clip_D1], feed_dict=FEED_DICT)
				# 	it_d1 += 1
				# 	d1_loss = sess.run(D1_loss, feed_dict=FEED_DICT)
				
				sess.run([D2_solver, clip_D2], feed_dict=FEED_DICT)
				it_d2 += 1
				d2_loss = sess.run(D2_loss, feed_dict=FEED_DICT)
				# while d2_loss > 1.38*n:
				# 	sess.run([D2_solver, clip_D2], feed_dict=FEED_DICT)
				# 	it_d2 += 1
				# 	d2_loss = sess.run(D2_loss, feed_dict=FEED_DICT)

				sess.run([G_solver, clip_G], feed_dict=FEED_DICT)
				it_g += 1
				g_loss_gan = sess.run(G_loss_GAN, feed_dict=FEED_DICT)

				# while g_loss_gan > 1.38*n:
				# 	sess.run([G_Content_solver,clip_G],feed_dict=FEED_DICT)
				# 	sess.run([G_solver, clip_G], feed_dict=FEED_DICT)
				# 	it_g += 1
				# 	g_loss_gan = sess.run(G_loss_GAN, feed_dict=FEED_DICT)
				g_loss_in = sess.run(LOSS_in, feed_dict=FEED_DICT)
				print('G_loss_in: ',g_loss_in)


				print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))

				# every 10 batch print the information about loss
				if batch % 10 == 0:
					elapsed_time = datetime.now() - start_time
					lr,g_loss, d1_loss, d2_loss = sess.run([learning_rate,G_loss, D1_loss, D2_loss], feed_dict=FEED_DICT)
					print('G_loss: %s, D1_loss: %s, D2_loss: %s ,Learing_rate: %s, selapsed_time: %s'  % (
						g_loss, d1_loss, d2_loss,lr,elapsed_time))

					g_loss_grad,g_loss_in,g_loss_content,g_loss_gan=sess.run([LOSS_grad,LOSS_in,G_loss_content,G_loss_GAN],feed_dict=FEED_DICT)
					print('LOSS_grad:%s, LOSS_in:%s, LOSS_content:%s, LOSS_GAN:%s'  % (
						g_loss_grad,g_loss_in,g_loss_content,g_loss_gan))

				# summary
				result = sess.run(merged, feed_dict=FEED_DICT)
				writer.add_summary(result, step)
				if step % logging_period == 0:
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')

	writer.close()
	saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	grad_img = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return grad_img







