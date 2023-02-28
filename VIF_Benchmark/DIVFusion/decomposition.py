# author:xxy,time:2022/2/22
############ tf的预定义 ############
from __future__ import print_function
import random
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
from glob import glob
import cv2
import losses
from model import *
############ 常量的预定义 ############
batch_size = 5
patch_size_x = 224
patch_size_y = 224


############ 变量的预定义 ############
sess = tf.InteractiveSession()
vi = tf.placeholder(tf.float32, [None, None, None, 1], name='vi')
vi_hist = tf.placeholder(tf.float32, [None, None, None, 1], name='vi_hist')
ir = tf.placeholder(tf.float32, [None, None, None, 1], name='ir')
vi_3 = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_3')
vi_hist_3 = tf.placeholder(tf.float32, [None, None, None, 3], name='vi_hist_3')
[ir_r, vi_e_r, l_r] = decomposition(vi, ir)  # 网络架构的函数在这里
vi_e_r_3 = tf.concat([vi_e_r, vi_e_r, vi_e_r], axis=3)


############ LOSS #############
def mutual_i_input_loss(input_I_low, input_im):  # 照度平滑度损失is
    input_gray = tf.image.rgb_to_grayscale(input_im)
    input_gray = input_im
    low_gradient_x = gradient(input_I_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
    low_gradient_y = gradient(input_I_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
    mut_loss = tf.reduce_mean(x_loss + y_loss)
    return mut_loss
def mutual_i_loss(input_I_low):  # 相互一致性损失
    low_gradient_x = gradient(input_I_low, "x")
    x_loss = (low_gradient_x)* tf.exp(-10*(low_gradient_x))
    low_gradient_y = gradient(input_I_low, "y")
    y_loss = (low_gradient_y) * tf.exp(-10*(low_gradient_y))
    mutual_loss = tf.reduce_mean(x_loss + y_loss)
    return mutual_loss


recon_loss_vi = tf.reduce_mean(tf.square(vi_e_r * l_r - vi))  # 重构损失可见光
recon_loss_ir = tf.reduce_mean(tf.square(ir_r - ir))  # 重构损失红外光
i_input_mutual_loss_low = mutual_i_input_loss(l_r, vi)  # 照度平滑度损失is
# 由于VGG适合3通道的图像，所以直接将原先预设的单通道图像复制成三通道的 vi_3,vi_hist_3
per_loss = losses.Perceptual_Loss(vi_e_r_3, vi_hist_3)  # 感知损失
mutual_loss = mutual_i_loss(l_r)  # 相互一致性损失
loss_Decom = 1000 * recon_loss_vi + 2000 * recon_loss_ir + 7 * i_input_mutual_loss_low + 40 * per_loss + 9 * mutual_loss


############ 训练预备 ############
lr = tf.placeholder(tf.float32, name='learning_rate')
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
train_op_Decom = optimizer.minimize(loss_Decom, var_list=var_Decom)
sess.run(tf.global_variables_initializer())
saver_Decom = tf.train.Saver(var_list=var_Decom)
print("[*] Initialize model successfully...")


############ 准备数据 ############
# load_data
train_ir_data = []
train_vi_data = []
train_vi_3_data = []
# train_vi_data_names = glob('D:\\Pycharm\\dataset\\train_vi\\*.jpg') #  专供测试代码时候用
# train_ir_data_names = glob('D:\\Pycharm\\dataset\\train_ir\\*.jpg') #  专供测试代码时候用
train_ir_data_names = glob('./ours_dataset_240/train/infrared/*.jpg') #  实际训练使用
train_vi_data_names = glob('./ours_dataset_240/train/visible/*.jpg')  #  实际训练使用
train_ir_data_names.sort()
train_vi_data_names.sort()
print('[*] Number of training data_ir/vi: %d' % len(train_ir_data_names))
for idx in range(len(train_ir_data_names)):
    im_before_ir = load_images(train_ir_data_names[idx])
    ir_gray = cv2.cvtColor(im_before_ir, cv2.COLOR_RGB2GRAY)
    train_ir_data.append(ir_gray)
    im_before_vi = load_images(train_vi_data_names[idx])
    vi_gray = cv2.cvtColor(im_before_vi, cv2.COLOR_RGB2GRAY)
    vi_y = rgb_ycbcr_np(im_before_vi)[:,:,0]
    train_vi_data.append(vi_y)  # 是归一化之后的图像形成一个list组
    vi_rgb = np.zeros_like(im_before_vi)  # 是为了和vgg匹配用的
    vi_rgb[:, :, 0] = vi_y
    vi_rgb[:, :, 1] = vi_y
    vi_rgb[:, :, 2] = vi_y
    train_vi_3_data.append(vi_rgb)
# eval_data
eval_ir_data = []
eval_vi_data = []
eval_vi_3_data = []
# eval_ir_data_name = glob('D:\\Pycharm\\dataset\\eval_ir\\*.jpg')
# eval_vi_data_name = glob('D:\\Pycharm\\dataset\\eval_vi\\*.jpg')
eval_ir_data_name = glob('./ours_dataset_240/test/infrared/*.jpg')
eval_vi_data_name = glob('./ours_dataset_240/test/visible/*.jpg')
eval_ir_data_name.sort()
eval_vi_data_name.sort()
for idx in range(len(eval_ir_data_name)):
    eval_im_before_ir = load_images(eval_ir_data_name[idx])
    eval_ir_gray = cv2.cvtColor(eval_im_before_ir, cv2.COLOR_RGB2GRAY)
    eval_ir_data.append(eval_ir_gray)
    eval_im_before_vi = load_images(eval_vi_data_name[idx])
    eval_vi_gray = cv2.cvtColor(eval_im_before_vi, cv2.COLOR_RGB2GRAY)
    eval_vi_y = rgb_ycbcr_np(eval_im_before_vi)[:,:,0]
    eval_vi_data.append(eval_vi_y)
    eval_vi_3 = np.zeros_like(eval_im_before_vi)  # 是为了和vgg匹配用的
    eval_vi_3[:, :, 0] = eval_vi_y
    eval_vi_3[:, :, 1] = eval_vi_y
    eval_vi_3[:, :, 2] = eval_vi_y
    eval_vi_3_data.append(eval_vi_3)


epoch = 2000
learning_rate = 0.0001
sample_dir_vi = './Decom_net_train_VI/'
if not os.path.isdir(sample_dir_vi):
    os.makedirs(sample_dir_vi)
sample_dir_ir = './Decom_net_train_IR/'
if not os.path.isdir(sample_dir_ir):
    os.makedirs(sample_dir_ir)

eval_every_epoch = 200
train_phase = 'decomposition'
numBatch = len(train_ir_data) // int(batch_size)  # 批数据量是10,一个小patch图片大小是48
train_op = train_op_Decom
train_loss = loss_Decom
saver = saver_Decom

checkpoint_dir = './checkpoint/decom_net_train/'
########
# checkpoint_dir = './checkpoint/decom_net_train_xiyan/'
########
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)



############ 训练开始~！ ############
start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
# epoch是2000
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):  # 总共的图片数目除以一个批数据10，所得的批数
        batch_input_ir = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype="float32")
        batch_input_vi = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype="float32")
        batch_input_vi_3 = np.zeros((batch_size, patch_size_y, patch_size_x, 3), dtype="float32")
        batch_input_vi_3_hist = np.zeros((batch_size, patch_size_y, patch_size_x, 3), dtype='float32')
        for patch_id in range(batch_size):
            # train_ir_data[image_id] = np.expand_dims(train_ir_data[image_id], -1)
            # train_vi_data[image_id] = np.expand_dims(train_vi_data[image_id], -1)
            train_ir_data[image_id] = np.reshape(train_ir_data[image_id], [1024, 1280, 1])
            train_vi_data[image_id] = np.reshape(train_vi_data[image_id], [1024, 1280, 1])
            h, w, _= train_ir_data[image_id].shape
            y = random.randint(0, h - patch_size_y - 1)
            #  返回参数1与参数2之间的任一整数，我草这也是个不错的处理方式，比浩哥的那个虽然不那么合理，但是简单
            x = random.randint(0, w - patch_size_x - 1)
            # rand_mode = random.randint(0, 7)
            # batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][y: y+patch_size_y, x: x+patch_size_x, :], rand_mode)
            batch_input_ir[patch_id, :, :, :] = train_ir_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            batch_input_vi[patch_id, :, :, :] = train_vi_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            batch_input_vi_3[patch_id, :, :, :] = train_vi_3_data[image_id][y: y + patch_size_y, x: x + patch_size_x, :]
            batch_input_vi_3_hist[patch_id, :, :, 0] = hist(batch_input_vi_3[patch_id, :, :, 0])
            batch_input_vi_3_hist[patch_id, :, :, 1] = hist(batch_input_vi_3[patch_id, :, :, 1])
            batch_input_vi_3_hist[patch_id, :, :, 2] = hist(batch_input_vi_3[patch_id, :, :, 2])
            image_id = (image_id + 1) % len(train_ir_data)
            # if image_id == 0:
            #     tmp = list(zip(train_low_data, train_high_data))  # 返回的列表长度被截断为最短的参数序列的长度
            #     random.shuffle(tmp)  # 重新洗牌的操作
            #     train_low_data, train_high_data = zip(*tmp)
        _, loss, loss_recon_vi, loss_recon_ir, loss_mutual, loss_per, loss_mutual_double = sess.run(
            [train_op, train_loss, recon_loss_vi, recon_loss_ir, i_input_mutual_loss_low, per_loss, mutual_loss],
            feed_dict={vi: batch_input_vi,\
                       ir: batch_input_ir,\
                       # vi_e_r_3: batch_input_vi_3,\
                       vi_hist_3: batch_input_vi_3_hist,\
                       lr: learning_rate})
        # input_low_hist: input_per_hist1
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        print("recon_vi:%.4f, recon_ir:%.4f, smooth:%.4f, per:%.4f, mutual:%.4f" \
              % (loss_recon_vi, loss_recon_ir, loss_mutual, loss_per, loss_mutual_double))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        # 训练了一段时间之后看当时epoch的结果
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        for idx in range(len(eval_vi_data)):
            # input_ir_eval = np.reshape(eval_ir_data[idx], [1, 1024, 1280, 1])
            # input_vi_eval = np.reshape(eval_vi_data[idx], [1, 1024, 1280, 1])
            # input_vi_eval = np.expand_dims(eval_vi_data[idx], axis=0)
            # input_ir_eval = np.expand_dims(eval_ir_data[idx], axis=0)
            input_vi_eval = np.expand_dims(eval_vi_data[idx], axis=[0, -1])
            input_ir_eval = np.expand_dims(eval_ir_data[idx], axis=[0, -1])
            result_1, result_2, result_3 = sess.run([vi_e_r, l_r, ir_r], feed_dict={vi: input_vi_eval, ir: input_ir_eval})
            save_images(os.path.join(sample_dir_vi, 'vi_%d_%d.png' % (idx + 1, epoch + 1)), result_1, result_2)
            save_images(os.path.join(sample_dir_ir, 'ir_%d_%d.png' % (idx + 1, epoch + 1)), result_3)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
print("[*] Finish training for phase %s." % train_phase)

