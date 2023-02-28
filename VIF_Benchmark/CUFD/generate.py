# Use a trained Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from dense_net import DenseFuseNet
from fusion import Strategy
from skimage import color
import copy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def generate(ir_path, vis_path, model_path_1,model_path_2, output_path=None):
    ir_img = imread(ir_path, flatten=True, mode='YCbCr') / 255.0
    vis_img = imread(vis_path, flatten=True, mode='YCbCr') / 255.0
    # ir_img = color.rgb2gray(ir_img)
    # vis_img = color.rgb2gray(vis_img)
    ir_dimension = list(ir_img.shape)
    vis_dimension = list(vis_img.shape)
    ir_dimension.insert(0, 1)
    ir_dimension.append(1)
    vis_dimension.insert(0, 1)
    vis_dimension.append(1)
    ir_img = ir_img.reshape(ir_dimension)
    vis_img = vis_img.reshape(vis_dimension)

    # g1 = tf.Graph()
    with tf.Graph().as_default(), tf.Session() as sess:
        print('I_encoder Begin!')
        vis_img = vis_img.astype(np.float32)
        ir_img = ir_img.astype(np.float32)
        dfn = DenseFuseNet("DenseFuseNet")
        f11,f12,f13,f14,f21,f22,f23,f24= dfn.transform_test_part1(vis_img, ir_img)
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=['m_encoder', 'DenseFuseNet/m_encoder'])
        part1_saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        part1_saver.restore(sess, model_path_1)

        f11 = sess.run(f11)
        f12 = sess.run(f12)
        f13 = sess.run(f13)
        f14 = sess.run(f14)
        f21 = sess.run(f21)
        f22 = sess.run(f22)
        f23 = sess.run(f23)
        f24 = sess.run(f24)

    with tf.Graph().as_default(),tf.Session() as sess:
        print('F_encoder Begin!')
        cf11 = copy.deepcopy(f11)
        cf12 = copy.deepcopy(f12)
        cf21 = copy.deepcopy(f21)
        cf22 = copy.deepcopy(f22)
        pf11 = tf.Variable(initial_value=cf11,dtype=tf.float32)
        pf12 = tf.Variable(initial_value=cf12, dtype=tf.float32)
        pf21 = tf.Variable(initial_value=cf21, dtype=tf.float32)
        pf22 = tf.Variable(initial_value=cf22, dtype=tf.float32)
        cf13 = copy.deepcopy(f13)
        cf14 = copy.deepcopy(f14)
        cf23 = copy.deepcopy(f23)
        cf24 = copy.deepcopy(f24)
        pf13 = tf.Variable(initial_value=cf13, dtype=tf.float32)
        pf14 = tf.Variable(initial_value=cf14, dtype=tf.float32)
        pf23 = tf.Variable(initial_value=cf23, dtype=tf.float32)
        pf24 = tf.Variable(initial_value=cf24, dtype=tf.float32)

        dfn = DenseFuseNet("DenseFuseNet")
        cu1, cu2, cu3, cu4 = dfn.transform_test_part2(pf11, pf12, pf13, pf14, pf21, pf22, pf23, pf24)
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=['s_encoder', 'DenseFuseNet/s_encoder'])
        part2_saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        part2_saver.restore(sess, model_path_2)

        cu1 = sess.run(cu1)
        cu2 = sess.run(cu2)
        cu3 = sess.run(cu3)
        cu4 = sess.run(cu4)

        c1, u1 = Strategy(cu1)
        c2, u2 = Strategy(cu2)
        c3, u3 = Strategy(cu3)
        c4, u4 = Strategy(cu4)


    with tf.Graph().as_default(), tf.Session() as sess:
        print('F_decoder Begin!')
        cc1 = copy.deepcopy(c1)
        cc2 = copy.deepcopy(c2)
        cc3 = copy.deepcopy(c3)
        cc4 = copy.deepcopy(c4)
        cou1 = copy.deepcopy(u1)
        cou2 = copy.deepcopy(u2)
        cou3 = copy.deepcopy(u3)
        cou4 = copy.deepcopy(u4)

        pc1 = tf.Variable(initial_value=cc1, dtype=tf.float32)
        pc2 = tf.Variable(initial_value=cc2, dtype=tf.float32)
        pc3 = tf.Variable(initial_value=cc3, dtype=tf.float32)
        pc4 = tf.Variable(initial_value=cc4, dtype=tf.float32)
        pu1 = tf.Variable(initial_value=cou1, dtype=tf.float32)
        pu2 = tf.Variable(initial_value=cou2, dtype=tf.float32)
        pu3 = tf.Variable(initial_value=cou3, dtype=tf.float32)
        pu4 = tf.Variable(initial_value=cou4, dtype=tf.float32)

        dfn = DenseFuseNet("DenseFuseNet")
        f1,f2,f3,f4 = dfn.transform_test_part3(pc1,pu1,pc2,pu2,pc3,pu3,pc4,pu4)
        sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=['s_decoder', 'DenseFuseNet/s_decoder'])
        part3_saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        part3_saver.restore(sess, model_path_2)

        f1=sess.run(f1)
        f2=sess.run(f2)
        f3=sess.run(f3)
        f4=sess.run(f4)

        f1 = f1 + 1*np.where(np.abs(f11) > np.abs(f21), f11, f21)
        f2 = f2 + 1*np.where(np.abs(f12) > np.abs(f22), f12, f22)
        f3 = f3 + 1*np.where(np.abs(f13) > np.abs(f23), f13, f23)
        f4 = f4 + 1*np.where(np.abs(f14) > np.abs(f24), f14, f24)

    with tf.Graph().as_default(), tf.Session() as sess:
        print('I_decoder Begin!')
        cf1 = copy.deepcopy(f1)
        cf2 = copy.deepcopy(f2)
        cf3 = copy.deepcopy(f3)
        cf4 = copy.deepcopy(f4)
        pf1 = tf.Variable(initial_value=cf1, dtype=tf.float32)
        pf2 = tf.Variable(initial_value=cf2, dtype=tf.float32)
        pf3 = tf.Variable(initial_value=cf3, dtype=tf.float32)
        pf4 = tf.Variable(initial_value=cf4, dtype=tf.float32)

        dfn = DenseFuseNet("DenseFuseNet")
        out_put = dfn.transform_test_part4(pf1,pf2,pf3,pf4)
        sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=['m_decoder', 'DenseFuseNet/m_decoder'])
        part4_saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.global_variables_initializer())
        part4_saver.restore(sess, model_path_1)

        output=sess.run(out_put)
        output = output[0, :, :,0]
        imsave(output_path, output)
