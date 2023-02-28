# Train the F_encoder, F_decoder networks

from __future__ import print_function
import cv2
import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage
import h5py
import random
import xlwt

from dense_net import DenseFuseNet
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

patch_size = 84
CHANNELS = 1  # gray scale, default

LEARNING_RATE = 0.002
EPSILON = 1e-9
DECAY_RATE = 0.9
eps = 1e-8


def train_part2(source_imgs, model_path, save_path, EPOCHES_set, BATCH_SIZE, logging_period=1):
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
        SOURCE_VIS = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE_VIS')
        SOURCE_IR = tf.placeholder(tf.float32, shape=(BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE_IR')
        dfn = DenseFuseNet("DenseFuseNet")
        f11, f12, f13, f14, f21, f22, f23, f24 = dfn.transform_test_part1(SOURCE_VIS, SOURCE_IR)
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=['m_encoder', 'DenseFuseNet/m_encoder'])
        part1_saver = tf.train.Saver(variables_to_restore)
        part1_saver.restore(sess, model_path)

        cu1,cu2,cu3,cu4,_f11,_f12,_f13,_f14,_f21,_f22,_f23,_f24= dfn.transform_recons_part2(f11,f12,f13,f14,f21,f22,f23,f24)
        i_f11 = tf.reduce_sum(_f11, axis=3)
        i_f11 = tf.reshape(i_f11, [84, 84, 84, 1])
        i_f21 = tf.reduce_sum(_f21, axis=3)
        i_f21 = tf.reshape(i_f21, [84, 84, 84, 1])

        if11 = tf.reduce_sum(f11, axis=3)
        if11 = tf.reshape(if11, [84, 84, 84, 1])
        if21 = tf.reduce_sum(f21, axis=3)
        if21 = tf.reshape(if21, [84, 84, 84, 1])

        loss_u1 = tf.reduce_mean((tf.minimum(abs(cu1[:, :, :, 0:32] - f11)
                                            , abs(cu1[:, :, :, 0:32] - f21))) ** 2
                                + (tf.minimum(abs(cu1[:, :, :, 32:64] - f11)
                                              , abs(cu1[:, :, :, 32:64] - f21))) ** 2
                                + 4 * tf.square(cu1[:, :, :, 0:32] - cu1[:, :, :, 32:64]))

        pixel_loss1 = tf.reduce_mean(7 * tf.square(_f11 - f11) + 3 * tf.square(_f21 - f21)
                                   + 18 * tf.square(grad(i_f11) - 1.5*grad(if11)) + 25 * tf.square(grad(i_f21) - 1.5*grad(if21)))

        loss_de1 = loss_u1 + 50 * pixel_loss1
        current_iter = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=current_iter,
                                                   decay_steps=int(n_batches), decay_rate=DECAY_RATE,
                                                   staircase=False)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op1 = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss_de1, global_step=current_iter)

        sess.run(tf.global_variables_initializer())

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list)

        tf.summary.scalar('de_loss1', loss_u1)
        tf.summary.scalar('de_loss2', pixel_loss1)
        tf.summary.scalar('de_loss', loss_de1)
        tf.summary.image('vis', SOURCE_VIS, max_outputs=3)
        tf.summary.image('ir', SOURCE_IR, max_outputs=3)

        tf.summary.image('f13', tf.expand_dims(f13[:,:,:,5],-1), max_outputs=3)
        tf.summary.image('fase_f13', tf.expand_dims(_f13[:,:,:,5],-1), max_outputs=3)
        tf.summary.image('f13_sum', tf.expand_dims(tf.reduce_sum(f13[:, :, :, :], -1),-1), max_outputs=3)
        tf.summary.image('fasef13_sum', tf.expand_dims(tf.reduce_sum(_f13[:, :, :, :], -1),-1), max_outputs=3)
        tf.summary.image('f23', tf.expand_dims(f23[:,:,:,5],-1), max_outputs=3)
        tf.summary.image('fase_f23', tf.expand_dims(_f23[:,:,:,5],-1), max_outputs=3)

        tf.summary.scalar('Learning rate', learning_rate)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        # ** Start Training **
        step = 0
        count_loss = 0

        for epoch in range(EPOCHS):
            num=0
            print('Train Epoch begin!')
            for batch in range(n_batches):
                step += 1
                current_iter = step
                VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
                IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
                VIS_batch = np.expand_dims(VIS_batch, -1)
                IR_batch = np.expand_dims(IR_batch, -1)
                FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch}
                # if batch<=100:
                #     FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch,
                #                  f21: t_f21_1[batch, :, :, :, :], f22: t_f22_1[batch, :, :, :, :],
                #                  f23: t_f23_1[batch, :, :, :, :], f24: t_f24_1[batch, :, :, :, :],
                #                  f11: t_f11_1[batch, :, :, :, :], f12: t_f12_1[batch, :, :, :, :],
                #                  f13: t_f13_1[batch, :, :, :, :], f14: t_f14_1[batch, :, :, :, :]}
                # elif batch <= 200:
                #     FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch,
                #                  f21: t_f21_add1[batch-101, :, :, :, :], f22: t_f22_add1[batch-101, :, :, :, :],
                #                  f23: t_f23_add1[batch-101, :, :, :, :], f24: t_f24_add1[batch-101, :, :, :, :],
                #                  f11: t_f11_add1[batch-101, :, :, :, :], f12: t_f12_add1[batch-101, :, :, :, :],
                #                  f13: t_f13_add1[batch-101, :, :, :, :], f14: t_f14_add1[batch-101, :, :, :, :]}
                # else:
                #     FEED_DICT = {SOURCE_VIS: VIS_batch, SOURCE_IR: IR_batch,
                #                  f21: t_f21_add2[batch-201, :, :, :, :], f22: t_f22_add2[batch-201, :, :, :, :],
                #                  f23: t_f23_add2[batch-201, :, :, :, :], f24: t_f24_add2[batch-201, :, :, :, :],
                #                  f11: t_f11_add2[batch-201, :, :, :, :], f12: t_f12_add2[batch-201, :, :, :, :],
                #                  f13: t_f13_add2[batch-201, :, :, :, :], f14: t_f14_add2[batch-201, :, :, :, :]}

                id = 0
                sess.run(train_op1, feed_dict=FEED_DICT)
                id += 1
                de_loss = sess.run(loss_de1, feed_dict=FEED_DICT)

                if batch % 2 == 0:
                    while de_loss > 1.4 and id < 20:
                        sess.run(train_op1, feed_dict=FEED_DICT)
                        id += 1


                if batch % 5 == 0:
                    de_loss = sess.run(loss_de1, feed_dict=FEED_DICT)
                    num=num+1
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('-----------------------------------------')
                    print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))
                    print('de_loss: %s\n' % (de_loss))
                    print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

                is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                result = sess.run(merged, feed_dict=FEED_DICT)
                writer.add_summary(result, step)

                if is_last_step or step % logging_period == 0:
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
                        epoch + 1, EPOCHS, step, lr, elapsed_time))
            writer.close()
            saver.save(sess, save_path + str(epoch + 1) + 'part2_model.ckpt')


def grad(img):
    kernel = tf.constant([[1/8, 1 / 8, 1/8], [1 / 8, -1, 1 / 8], [1/8, 1 / 8, 1/8]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g


