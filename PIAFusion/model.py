import os
import time
import numpy as np
import h5py
import tensorflow as tf
from ops import *
from train_network import PIAFusion, Illumination_classifier
from tqdm import tqdm
from natsort import natsorted
from PIL import Image

from utils import *

PIAfusion_net = PIAFusion()
IC_net = Illumination_classifier()


class PIAFusion(object):
    def __init__(self,
                 sess,
                 image_size=132,
                 label_size=120,
                 batch_size=32,
                 checkpoint_dir=None,
                 model_type=None,
                 phase=None,
                 Data_set=None,
                 Method=None,
                 ir_dir=None, 
                 vi_dir=None,
                 save_dir = None,
                 ):

        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.phase = phase
        self.DataSet = Data_set
        self.ir_dir = ir_dir
        self.Method = Method
        self.vi_dir = vi_dir
        self.save_dir = save_dir

    def build_classifier_model(self):
        with tf.name_scope('input'):
            # image patch
            self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3],
                                                   name='images')
            self.labels = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.label_size], name='label')

        with tf.compat.v1.variable_scope('classifier', reuse=False):
            self.predicted_label = IC_net.illumination_classifier(self.images, reuse=False)

        with tf.name_scope("learn_rate"):
            self.lr = tf.placeholder(tf.float32, name='lr')

        with tf.name_scope('c_loss'):
            self.classifier_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.predicted_label, labels=self.labels))
            # Operation comparing prediction with true label
            correct_prediction = tf.equal(tf.argmax(self.predicted_label, 1), tf.argmax(self.labels, 1))

            # Operation calculating the accuracy of our predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.compat.v1.summary.scalar which is used to display scalar information
            # used to display loss
            tf.compat.v1.summary.scalar('classifier loss', self.classifier_loss)
            self.c_loss_total = 10 * self.classifier_loss
            # display total_loss
            tf.compat.v1.summary.scalar('loss_c', self.c_loss_total)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)
        with tf.name_scope('image'):
            tf.compat.v1.summary.image('image', self.images[0:1, :, :, 0:3])

    def initial_classifier_model(self, Illum_images):
        with tf.compat.v1.variable_scope('classifier', reuse=False):
            self.predicted_label = IC_net.illumination_classifier(Illum_images, reuse=False)
        self.Illum_saver = tf.compat.v1.train.Saver(max_to_keep=50)

    def build_PIAFusion_model(self):
        with tf.name_scope('input'):
            # Visible image patch
            self.vi_images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3], name='vi_images')
            self.ir_images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1], name='ir_images')
            self.Y_images, self.Cb_images, self.Cr_images = RGB2YCbCr(self.vi_images)

        with tf.name_scope("learn_rate"):
            self.lr = tf.placeholder(tf.float32, name='lr')
        tf.global_variables_initializer().run()
        self.initial_classifier_model(self.vi_images)
        with tf.compat.v1.variable_scope('PIAFusion', reuse=False):
            self.fused_images, self.vi_features, self.ir_features = PIAfusion_net.PIAFusion(self.Y_images, self.ir_images, reuse=False)
        self.RGB_fused_images = YCbCr2RGB(self.fused_images, self.Cb_images, self.Cr_images, mode=1)
        print(self.checkpoint_dir)
        could_load = self.load(self.Illum_saver, self.checkpoint_dir, model_dir="%s" % ("Illumination"))
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        with tf.compat.v1.variable_scope('classifier', reuse=True):
            self.predicted_label = IC_net.illumination_classifier(self.vi_images, reuse=True)
        day_probability = self.predicted_label[:, 0]
        night_probability = self.predicted_label[:, 1]
        self.vi_w, self.ir_w = illumination_mechanism(day_probability, night_probability)
        self.vi_w = tf.reshape(self.vi_w, shape=[self.batch_size, 1, 1, 1])
        self.ir_w = tf.reshape(self.ir_w, shape=[self.batch_size, 1, 1, 1])
        with tf.name_scope('grad_bin'):
            self.Image_vi_grad = gradient(self.Y_images)
            self.Image_ir_grad = gradient(self.ir_images)
            self.Image_fused_grad = gradient(self.fused_images)
            self.Image_max_grad = tf.round((self.Image_vi_grad + self.Image_ir_grad) // (
                    tf.abs(self.Image_vi_grad + self.Image_ir_grad) + 0.0000000001)) * tf.maximum(
                tf.abs(self.Image_vi_grad), tf.abs(self.Image_ir_grad))
            self.concat_images = tf.concat([self.ir_images, self.Y_images], axis=-1, )
            self.pseudo_images = 0.7 * tf.reduce_max(self.concat_images, axis=-1, keepdims=True) + 0.3 * (tf.multiply(self.vi_w, self.Y_images)  + tf.multiply(self.ir_w, self.ir_images))

            self.RGB_pseudo_images = YCbCr2RGB(self.pseudo_images, self.Cb_images, self.Cr_images, mode=1)
        with tf.name_scope('f_loss'):
            self.ir_l1_loss = tf.reduce_mean(tf.abs(self.fused_images - self.ir_images))
            self.vi_l1_loss = tf.reduce_mean(tf.abs(self.fused_images - self.Y_images))
            self.ir_grad_loss = tf.reduce_mean(tf.abs(self.Image_fused_grad - self.Image_ir_grad))
            self.vi_grad_loss = tf.reduce_mean(tf.abs(self.Image_fused_grad - self.Image_vi_grad))
            self.joint_grad_loss = L1_loss(self.Image_fused_grad, self.Image_max_grad)
            self.pixel_loss = L1_loss(self.pseudo_images, self.fused_images)
            self.f_total_loss = 50 * self.pixel_loss + 50 * self.joint_grad_loss

            tf.compat.v1.summary.scalar('IR L1 loss', self.ir_l1_loss)
            tf.compat.v1.summary.scalar('VI L1 loss', self.vi_l1_loss)
            tf.compat.v1.summary.scalar('IR Gradient loss', self.ir_grad_loss)
            tf.compat.v1.summary.scalar('Fusion model total loss', self.f_total_loss)
            tf.compat.v1.summary.scalar('VI Gradient loss', self.vi_grad_loss)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=50)

        with tf.name_scope('image'):
            tf.compat.v1.summary.image('ir_image', self.ir_images[0:1, :, :, :])
            tf.compat.v1.summary.image('vi_image', self.vi_images[0:1, :, :, :])
            tf.compat.v1.summary.image('fused image', self.RGB_fused_images[0:1, :, :, :])
            tf.compat.v1.summary.image('pseudo images', self.RGB_pseudo_images[0:1, :, :, :])
            tf.compat.v1.summary.image('ir_feature', self.ir_features[0:1, :, :, 0:1])
            tf.compat.v1.summary.image('vi_feature', self.vi_features[0:1, :, :, 0:1])
            tf.compat.v1.summary.image('joint_gradient', self.Image_max_grad[0:1, :, :, 0:1])
            tf.compat.v1.summary.image('fused_gradient', self.Image_fused_grad[0:1, :, :, 0:1])


    def train(self, config):
        variables_dir = './variables'
        check_folder(variables_dir)
        variables_name = os.path.join(variables_dir, self.model_type + '.txt')
        if os.path.exists(variables_name):
            os.remove(variables_name)
        if config.model_type == 'Illum':
            print('train Illumination classifier!')
            print('Data Preparation!~')
            dataset_name = 'data_illum.h5'
            f = h5py.File(dataset_name, 'r')
            sources = f['data'][:]
            print(sources.shape)
            sources = np.transpose(sources, (0, 3, 2, 1))
            images = sources[:, :, :, 0:3]
            labels = sources[:, 0, 0, 3:5]
            images = images ## input image [0, 1]
            self.build_classifier_model()
            num_imgs = sources.shape[0]
            # num_imgs = 800
            mod = num_imgs % self.batch_size
            n_batches = int(num_imgs // self.batch_size)
            print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
            self.iteration = n_batches
            if mod > 0:
                print('Train set has been trimmed %d samples...\n' % mod)
                sources = sources[:-mod]
            print("source shape:", sources.shape)
            batch_idxs = n_batches
            tensorboard_path, log_path = form_results(dataset=config.DataSet, model_type=self.model_type)
            t_vars = tf.trainable_variables()
            C_vars = [var for var in t_vars if 'Classifier' in var.name]
            log_name = log_path + '/log.txt'
            if os.path.exists(log_name):
                os.remove(log_name)
            for var in C_vars:
                with open(variables_name, 'a') as log:
                    log.write(var.name)
                    log.write('\n')

            self.C_vars = C_vars
            with tf.name_scope('train_step'):
                self.train_classifier_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.c_loss_total,
                                                                                                 var_list=self.C_vars)

            self.summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)
            tf.initialize_all_variables().run()
            counter = 0
            start_time = time.time()
            total_classifier_loss = 0
            total_loss = 0
            total_accuracy = 0
            show_num = 5
            show_count = 0
            if config.is_train:
                self.init_lr = config.learning_rate
                self.decay_epoch = int(config.epoch / 2)
                print("Training...")
                for ep in range(config.epoch):
                    # Run by batch images
                    lr = self.init_lr if ep < self.decay_epoch else self.init_lr * (config.epoch - ep) / (
                            config.epoch - self.decay_epoch)  # linear decay
                    batch_idxs = batch_idxs
                    for idx in range(0, batch_idxs):
                        batch_images = images[idx * config.batch_size: (idx + 1) * config.batch_size]
                        batch_labels = labels[idx * config.batch_size: (idx + 1) * config.batch_size]
                        counter += 1
                        _, err_g, batch_classifer_loss, batch_accuracy, summary_str, predicted_label = self.sess.run(
                            [self.train_classifier_op, self.c_loss_total, self.classifier_loss, self.accuracy,
                             self.summary_op, self.predicted_label],
                            feed_dict={self.images: batch_images, self.labels: batch_labels, self.lr: lr})
                        # Write the statistics to the log file
                        total_classifier_loss += batch_classifer_loss
                        total_loss += err_g
                        total_accuracy += batch_accuracy
                        show_count += 1
                        writer.add_summary(summary_str, global_step=counter)
                        if idx % show_num == show_num - 1:
                            print("learn rate:[%0.6f]" % (lr))
                            print(
                                "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], classifier_loss:[%.4f], accuracy:[%.4f]"
                                % ((ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                                   total_loss / show_count, total_classifier_loss / show_count,
                                   total_accuracy / show_count))
                            # print(predicted_label)
                            with open(log_path + '/log.txt', 'a') as log:
                                log.write(
                                    "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], classifier_loss:[%.4f], accuracy:[%.4f] \n"
                                    % ((ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                                       total_loss / show_count, total_classifier_loss / show_count,
                                       total_accuracy / show_count))
                            total_classifier_loss = 0
                            total_loss = 0
                            total_accuracy = 0
                            show_count = 0
                            start_time = time.time()
                    self.save(config.checkpoint_dir, ep)
        else:
            print(self.model_type == 'PIAFusion')
            print("Data preparation!")
            dataset_name = 'data_VIF.h5'
            # if config.DataSet == 'TNO':
            #     dataset_name = 'data_VIF.h5'
            # elif config.DataSet == 'RoadScene':
            #     dataset_name = 'data_road.h5'
            f = h5py.File(dataset_name, 'r')
            sources = f['data'][:]
            print(sources.shape)
            sources = np.transpose(sources, (0, 3, 2, 1))
            images = sources
            images = images
            self.build_PIAFusion_model()
            if config.is_train:
                print('images shape: ', images.shape)
                num_imgs = sources.shape[0]
                mod = num_imgs % self.batch_size
                n_batches = int(num_imgs // self.batch_size)
                print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
                self.iteration = n_batches
                if mod > 0:
                    print('Train set has been trimmed %d samples...\n' % mod)
                    sources = sources[:-mod]
                print("source shape:", sources.shape)
                batch_idxs = n_batches
            tensorboard_path, log_path = form_results(dataset=config.DataSet, model_type=self.model_type)
            t_vars = tf.trainable_variables()
            f_vars = [var for var in t_vars if 'classifier' not in var.name]
            log_name = log_path + '/log.txt'
            if os.path.exists(log_name):
                os.remove(log_name)
            for var in f_vars:
                with open(variables_name, 'a') as log:
                    log.write(var.name)
                    log.write('\n')

            self.f_vars = f_vars
            with tf.name_scope('train_step'):
                self.train_iafusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.f_total_loss,
                                                                                               var_list=self.f_vars)
            I_vars = tf.global_variables()
            I_vars = [var for var in I_vars if 'classifier' not in var.name]
            init = tf.variables_initializer(I_vars)
            self.sess.run(init)
            self.summary_op = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)
            counter = 0
            start_time = time.time()
            total_ir_l1_loss = 0
            total_vi_l1_loss = 0
            total_ir_grad_loss = 0
            total_vi_grad_loss = 0
            total_loss = 0
            show_num = 5
            show_count = 0
            self.init_lr = config.learning_rate
            self.decay_epoch = int(config.epoch / 2)
            print("Training...")
            print(self.sess.run(tf.get_default_graph().get_tensor_by_name("classifier/Classifier/conv1/bias:0")))
            for ep in range(config.epoch):
                # Run by batch images
                lr = self.init_lr if ep < self.decay_epoch else self.init_lr * (config.epoch - ep) / (
                        config.epoch - self.decay_epoch)  # linear decay
                batch_idxs = batch_idxs
                for idx in range(0, batch_idxs):
                    batch_images = images[idx * config.batch_size: (idx + 1) * config.batch_size]
                    vi_batch_images = batch_images[:, :, :, 0:3]
                    ir_batch_images = batch_images[:, :, :, 3:4]
                    counter += 1
                    _, err_g, ir_batch_l1_loss, vi_batch_l1_loss, ir_batch_grad_loss, vi_batch_grad_loss, vi_batch_w, ir_batch_w, summary_str = self.sess.run(
                        [self.train_iafusion_op, self.f_total_loss, self.ir_l1_loss, self.vi_l1_loss, self.ir_grad_loss,
                         self.vi_grad_loss, self.vi_w, self.ir_w, self.summary_op],
                        feed_dict={self.ir_images:ir_batch_images, self.vi_images:vi_batch_images, self.lr:lr})

                    # Write the statistics to the log file
                    total_ir_l1_loss += ir_batch_l1_loss
                    total_vi_l1_loss += vi_batch_l1_loss
                    total_ir_grad_loss += ir_batch_grad_loss
                    total_vi_grad_loss += vi_batch_grad_loss
                    total_loss += err_g
                    show_count += 1
                    writer.add_summary(summary_str, global_step=counter)
                    if idx % show_num == show_num - 1:
                        print("learn rate:[%0.6f]" % (lr))
                        print(
                            "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], ir_L1_loss:[%.4f], vi_L1_loss:[%.4f], ir_gradient_loss:[%.4f], vi_gradient_loss:[%.4f], vi_weight:[%.4f], ir_weight:[%.4f]"
                            % ((ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                               err_g, ir_batch_l1_loss, vi_batch_l1_loss,
                               ir_batch_grad_loss, vi_batch_grad_loss, vi_batch_w[1], ir_batch_w[1]))
                        print('vi_weight:', vi_batch_w[10], ', ir_weight:', ir_batch_w[10])
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write(
                                "Epoch:[%d/%d], step:[%d/%d], time: [%4.4f], loss_g:[%.4f], ir_L1_loss:[%.4f], vi_L1_loss:[%.4f], ir_gradient_loss:[%.4f], vi_gradient_loss:[%.4f]\n"
                                % ((ep + 1), config.epoch, idx + 1, batch_idxs, time.time() - start_time,
                               err_g, ir_batch_l1_loss, vi_batch_l1_loss,
                               ir_batch_grad_loss, vi_batch_grad_loss))
                        counter = 0
                        total_ir_l1_loss = 0
                        total_vi_l1_loss = 0
                        total_ir_grad_loss = 0
                        total_vi_grad_loss = 0
                        total_illumination_loss = 0
                        total_loss = 0
                        show_num = 5
                        show_count = 0
                        start_time = time.time()
                self.save(config.checkpoint_dir, ep)

    def test(self, config):
        if self.model_type == 'Illum':
            test_day_dir = './test_data/Illum/day'
            test_night_dir = './/test_data/Illum/night'
            with tf.name_scope('input'):
                # infrared image patch
                self.images = tf.placeholder(tf.float32, [1, None, None, 3], name='images')
            self.initial_classifier_model(self.images)
            tf.global_variables_initializer().run()
            print(self.checkpoint_dir)
            could_load = self.load(self.Illum_saver, self.checkpoint_dir)
            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            filelist = os.listdir(test_day_dir)
            # filelist.sort(key=lambda x: int(x[0:-4]))
            with tf.compat.v1.variable_scope('classifier', reuse=True):
                self.predicted_label = IC_net.illumination_classifier(self.images, reuse=True)
            True_count = 0
            Total_count = 0
            for item in filelist:
                test_day_file = os.path.join(os.path.abspath(test_day_dir), item)
                test_day_image = load_test_data(test_day_file, mode=2)
                test_day_image = np.asarray(test_day_image)
                print('test_day_image:', test_day_image.shape)
                predicted_label = self.sess.run(self.predicted_label, feed_dict={self.images: test_day_image})
                correct_prediction = np.argmax(predicted_label, 1)
                if correct_prediction[0] == 0:
                    True_count += 1
                Total_count += 1
                print('input: {}, predicted_label: {}, correct_prediction: {}'.format('ir image', predicted_label,
                                                                                      correct_prediction))
            filelist = os.listdir(test_night_dir)
            # filelist.sort(key=lambda x: int(x[0:-4]))
            for item in filelist:
                test_night_file = os.path.join(os.path.abspath(test_night_dir), item)
                test_night_image = load_test_data(test_night_file, mode=2)
                test_night_image = np.asarray(test_night_image)
                print('test_night_image:', test_night_image.shape)
                predicted_label = self.sess.run(self.predicted_label, feed_dict={self.images: test_night_image})
                correct_prediction = np.argmax(predicted_label, 1)
                if correct_prediction[0] == 1:
                    True_count += 1
                Total_count += 1
                print('input: {}, predicted_label: {}, correct_prediction: {}'.format('ir image', predicted_label,
                                                                                      correct_prediction))
            print('Testing Ending, Testing number is {}, Testing accuracy is {:.2f}%'.format(Total_count,
                                                                                             True_count / Total_count * 100))
        else:
            
            self.build_PIAFusion_model()
            tf.global_variables_initializer().run()
            print(self.checkpoint_dir)
            could_load = self.load(self.saver, self.checkpoint_dir)

            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            filelist = natsorted(os.listdir(self.ir_dir))
            check_folder(self.save_dir)

            with tf.name_scope('input'):
                # infrared image patch
                self.ir_images = tf.placeholder(tf.float32, [1, None, None, 1], name='ir_images')
                self.vi_images = tf.placeholder(tf.float32, [1, None, None, 3], name='vi_images')
                self.Y_images, self.Cb_images, self.Cr_images = RGB2YCbCr(self.vi_images)
                # print(self.Y_images.get_shape().as_list())


            with tf.compat.v1.variable_scope('PIAFusion', reuse=False):
                self.fused_images = PIAfusion_net.PIAFusion(self.Y_images, self.ir_images, reuse=True, Feature_out=False)
                self.RGB_fused_images = YCbCr2RGB(self.fused_images, self.Cb_images, self.Cr_images, mode=2)
            time_list = []
            test_bar = tqdm(filelist)
            for item in test_bar:
                test_ir_file = os.path.join(os.path.abspath(self.ir_dir), item)
                test_vi_file = os.path.join(os.path.abspath(self.vi_dir), item)
                self.fusion_path = os.path.join(self.save_dir, item)
                test_ir_image = load_test_data(test_ir_file)
                test_vi_image = load_test_data(test_vi_file, mode=2)
                test_ir_image = np.asarray(test_ir_image)
                test_vi_image = np.asarray(test_vi_image)
                start = time.time()
                fused_image = self.sess.run(
                    self.RGB_fused_images,
                    feed_dict={self.ir_images: test_ir_image, self.vi_images: test_vi_image})
                fused_image = fused_image.squeeze()
                fused_image = fused_image * 255.0
                end = time.time()
                cv2.imwrite(self.fusion_path, fused_image)                         
                test_bar.set_description('{} | {} {:.4f}'.format(self.Method, item, end-start))

    def save(self, checkpoint_dir, step):
        print(self.model_type)
        if self.model_type == 'Illum':
            model_name = 'Illumination.model'
            model_dir = "%s" % ("Illumination")
        elif self.model_type =='PIAFusion':
            model_name = "IAFusion.model"
            model_dir = "%s" % ("PIAFusion")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        check_folder(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, saver, checkpoint_dir, model_dir=None):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
