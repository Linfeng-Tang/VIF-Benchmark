# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage
from scipy.misc import imread

from dense_net import DenseFuseNet

patch_size = 84
CHANNELS = 1 # gray scale, default

LEARNING_RATE = 0.0002
EPSILON = 1e-5
DECAY_RATE = 0.9
eps = 1e-8


def train_part1(source_imgs, save_path, EPOCHES_set, BATCH_SIZE, logging_period=1):
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
        SOURCE = tf.placeholder(tf.float32, shape=(2*BATCH_SIZE, patch_size, patch_size, 1), name='SOURCE')

        dfn = DenseFuseNet("DenseFuseNet")
        f1, f2, f3, f4, _f = dfn.transform_recons_part1(SOURCE)

        loss_en = tf.reduce_mean(1*tf.square(_f - SOURCE) + 40 * tf.square(grad(_f) - grad(SOURCE)))

        current_iter = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=current_iter,
                                                   decay_steps=int(n_batches), decay_rate=DECAY_RATE,
                                                   staircase=False)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss_en,global_step=current_iter)
        sess.run(tf.global_variables_initializer())

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list)

        tf.summary.scalar('en_loss', loss_en)
        tf.summary.image('vis', SOURCE[0:84,:,:,:], max_outputs=3)
        tf.summary.image('ir', SOURCE[84:168, :, :, :], max_outputs=3)
        tf.summary.image('fase_vis', _f[0:84,:,:,:], max_outputs=3)
        tf.summary.image('fase_ir', _f[84:168, :, :, :], max_outputs=3)

        tf.summary.scalar('Learning rate', learning_rate)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        # ** Start Training **
        step = 0
        count_loss = 0

        for epoch in range(EPOCHS):
            np.random.shuffle(source_imgs)
            print('Train Epoch begin!')
            for batch in range(n_batches):
                step += 1
                current_iter = step
                VIS_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 1]
                VIS_batch = (VIS_batch-np.min(VIS_batch))/(np.max(VIS_batch)-np.min(VIS_batch))
                IR_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0]
                IR_batch = (IR_batch - np.min(IR_batch)) / (np.max(IR_batch) - np.min(IR_batch))
                VIS_batch = np.expand_dims(VIS_batch, -1)
                IR_batch = np.expand_dims(IR_batch, -1)
                source = np.concatenate((VIS_batch,IR_batch),axis=0)
                FEED_DICT = {SOURCE:source}

                sess.run(train_op, feed_dict=FEED_DICT)
                id_1=0
                sess.run(train_op, feed_dict=FEED_DICT)
                id_1 += 1
                en_loss = sess.run(loss_en, feed_dict=FEED_DICT)

                if batch % 2 == 0:
                    while en_loss > 0.01 and id_1 < 10:
                        sess.run(train_op, feed_dict=FEED_DICT)
                        en_loss = sess.run(loss_en, feed_dict=FEED_DICT)
                        id_1 += 1

                if batch % 5 == 0:
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('-----------------------------------------')
                    print("epoch: %d/%d, batch: %d\n" % (epoch + 1, EPOCHS, batch))
                    print('en_loss: %s' % (en_loss))
                    print("lr: %s, elapsed_time: %s\n" % (lr, elapsed_time))

                result = sess.run(merged, feed_dict=FEED_DICT)
                writer.add_summary(result, step)

                is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)
                if is_last_step or step % logging_period == 0:
                    elapsed_time = datetime.now() - start_time
                    lr = sess.run(learning_rate)
                    print('epoch:%d/%d, step:%d, lr:%s, elapsed_time:%s' % (
                        epoch + 1, EPOCHS, step, lr, elapsed_time))
            writer.close()
            saver.save(sess, save_path + str(epoch + 1) + 'part1_model.ckpt')

def grad(img):
    # kernel = tf.constant([[1/8, 1 / 8, 1/8], [1 / 8, -1, 1 / 8], [1/8, 1 / 8, 1/8]])
    kernel = tf.constant([[0, 1 / 4, 0], [1 / 4, -1, 1 / 4], [0, 1 /4, 0]])
    kernel = tf.expand_dims(kernel, axis=-1)
    kernel = tf.expand_dims(kernel, axis=-1)
    g = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return g











