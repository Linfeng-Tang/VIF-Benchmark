import os
import time
import numpy as np
import tensorflow as tf
from train_network import STDFusionNet
from utils import (
    read_data,
    input_setup,
    imsave,
    merge,
    gradient
)

STDFusion_net = STDFusionNet()


class STDFusion(object):
    def __init__(self,
                 sess,
                 image_size=132,
                 label_size=120,
                 batch_size=32,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            # Visible image patch
            self.ir_images = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='ir_images')
            self.vi_images = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='vi_images')
            self.ir_mask = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='ir_mask')

        with tf.name_scope('Fusion'):
            self.fusion_images = STDFusion_net.STDFusion_model(self.vi_images, self.ir_images)
        with tf.name_scope("learn_rate"):
            self.lr = tf.placeholder(tf.float32, name='lr')

        with tf.name_scope('g_loss'):
            self.ir_mask = (self.ir_mask + 1) / 2.0
            self.ir_p_loss_train = tf.multiply(self.ir_mask, tf.abs(self.fusion_images - self.ir_images))
            self.vi_p_loss_train = tf.multiply(1 - self.ir_mask, tf.abs(self.fusion_images - self.vi_images))
            self.ir_grad_loss_train = tf.multiply(self.ir_mask, tf.abs(gradient(self.fusion_images) - gradient(self.ir_images)))
            self.vi_grad_loss_train = tf.multiply(1 - self.ir_mask, tf.abs(gradient(self.fusion_images) - gradient(self.vi_images)))

            self.ir_p_loss = tf.reduce_mean(self.ir_p_loss_train)
            self.vi_p_loss = tf.reduce_mean(self.vi_p_loss_train)
            self.ir_grad_loss = tf.reduce_mean(self.ir_grad_loss_train)
            self.vi_grad_loss = tf.reduce_mean(self.vi_grad_loss_train)
            self.g_loss_2 = 1 * self.vi_p_loss + 1 * self.vi_grad_loss + 7 * self.ir_p_loss + 7 * self.ir_grad_loss

            # tf.compat.v1.summary.scalar which is used to display scalar information
            # used to display loss
            tf.compat.v1.summary.scalar('g_loss_2', self.g_loss_2)
            self.g_loss_total = 1 * self.g_loss_2
            # display total_loss
            tf.compat.v1.summary.scalar('loss_g', self.g_loss_total)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)

        with tf.name_scope('image'):
            tf.compat.v1.summary.image('vi_image', tf.expand_dims(self.vi_images[1, :, :, :], 0))
            tf.compat.v1.summary.image('ir_image', tf.expand_dims(self.ir_images[1, :, :, :], 0))
            tf.compat.v1.summary.image('fusion_images', tf.expand_dims(self.fusion_images[1, :, :, :], 0))

    def form_results(self, results_path='./Results'):
        """
        Forms folders for each run to store the tensorboard files, saved models and the log files.
        :return: three string pointing to tensorboard, saved models and log paths respectively.
        """
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        folder_name = "/{0}_{1}_{2}_model". \
            format('STDFusion', self.batch_size, 'Pixel_Grad')
        tensorboard_path = results_path + folder_name + '/Tensorboard'
        saved_model_path = results_path + folder_name + '/Saved_models/'
        log_path = results_path + folder_name + '/log'
        if not os.path.exists(results_path + folder_name):
            os.mkdir(results_path + folder_name)
            os.mkdir(tensorboard_path)
            os.mkdir(saved_model_path)
            os.mkdir(log_path)
        return tensorboard_path, saved_model_path, log_path

    def train(self, config):

        if config.is_train:
            print("Data preparation!")
            input_setup(self.sess, config, "Train_ir")
            input_setup(self.sess, config, "Train_vi")
            input_setup(self.sess, config, 'Train_ir_mask_blur')
        """
        else:
          nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
          nx_vi,ny_vi=input_setup(self.sess, config,"Test_vi")
        """
        if config.is_train:
            data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir", "train.h5")
            # print(data_dir_ir)
            data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi", "train.h5")
            train_data_ir_mask = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir_mask_blur", "train.h5")
        else:
            data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Test_ir", "test.h5")
            data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Test_vi", "test.h5")

        print("Data preparation over!")
        print("Reading data!")
        train_data_ir = read_data(data_dir_ir)
        train_data_vi = read_data(data_dir_vi)
        train_data_ir_mask = read_data(train_data_ir_mask)
        t_vars = tf.trainable_variables()
        # print(t_vars)
        for var in t_vars:
            with open('variables.txt', 'a') as log:
                log.write(var.name)
                log.write('\n')

        self.g_vars = t_vars
        with tf.name_scope('train_step'):
            self.train_generator_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total, var_list=self.g_vars)

        self.summary_op = tf.summary.merge_all()
        tensorboard_path, saved_model_path, log_path = self.form_results()
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)

        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()
        total_vi_p_loss = 0
        total_ir_p_loss = 0
        total_vi_grad_loss = 0
        total_ir_grad_loss = 0
        total_loss = 0
        show_num = 50
        if config.is_train:
            self.init_lr = config.learning_rate
            self.decay_epoch = int(config.epoch / 2)

            print("Training...")
            for ep in range(config.epoch):
                # Run by batch images
                lr = self.init_lr if ep < self.decay_epoch else self.init_lr * (config.epoch - ep) / (
                        config.epoch - self.decay_epoch)  # linear decay
                batch_idxs = len(train_data_ir) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_vi_images = train_data_vi[idx * config.batch_size: (idx + 1) * config.batch_size]
                    # print(np.size(batch_ir_images, 0))
                    batch_ir_images = train_data_ir[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_ir_mask = train_data_ir_mask[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_ir_mask = (batch_ir_mask + 1.0) / 2.0

                    counter += 1
                    for generator_num in range(1):

                        _, err_g, batch_vi_p_loss, batch_ir_p_loss, batch_vi_grad_loss, batch_ir_grad_loss, \
                        summary_str = self.sess.run(
                            [self.train_generator_op, self.g_loss_total, self.vi_p_loss, self.ir_p_loss,
                             self.vi_grad_loss, self.ir_grad_loss,
                             self.summary_op],
                            feed_dict={self.vi_images: batch_vi_images, self.ir_images: batch_ir_images,
                                       self.ir_mask: batch_ir_mask, self.lr:lr})
                    # Write the statistics to the log file
                    total_vi_p_loss += batch_vi_p_loss
                    total_ir_p_loss += batch_ir_p_loss
                    total_vi_grad_loss += batch_vi_grad_loss
                    total_ir_grad_loss += batch_ir_grad_loss
                    total_loss += err_g
                    writer.add_summary(summary_str, global_step=counter)
                    # self.train_writer.add_summary(summary_str, counter)

                    if idx % show_num == show_num - 1:
                        print("learn rate:[%0.6f]" % (lr))
                        print(
                            "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], vi_p_loss:[%.4f],"
                            " ir_p_loss:[%.4f], vi_grad_loss:[%0.4f], ir_grad_loss:[%0.4f]" % (
                                (ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                                total_loss / show_num,
                                total_vi_p_loss / show_num, total_ir_p_loss / show_num, total_vi_grad_loss / show_num,
                                total_ir_grad_loss / show_num))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write(
                                "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], vi_p_loss:[%.4f],"
                                " ir_p_loss:[%.4f], vi_grad_loss:[%0.4f], ir_grad_loss:[%0.4f]\n" % (
                                    (ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                                    total_loss / show_num,
                                    total_vi_p_loss / show_num, total_ir_p_loss / show_num,
                                    total_vi_grad_loss / show_num,
                                    total_ir_grad_loss / show_num))
                        total_vi_p_loss = 0
                        total_ir_p_loss = 0
                        total_vi_grad_loss = 0
                        total_ir_grad_loss = 0
                        total_loss = 0
                        start_time = time.time()
                self.save(config.checkpoint_dir, ep)
        else:
            print("Testing...")
            result = self.fusion_images.eval(
                feed_dict={self.ir_images: batch_ir_images, self.vi_images: batch_vi_images})
            result = result * 127.5 + 127.5
            result = merge(result, [nx_ir, ny_ir])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            imsave(result, image_path)

    def save(self, checkpoint_dir, step):
        model_name = "Fusion.model"
        model_dir = "%s_%s_%s" % ("STDFusion", self.batch_size, "Pixel_Grad")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s" % ("STDFusion", self.label_size, "Pixel_Grad")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
