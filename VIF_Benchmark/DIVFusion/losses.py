# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import time
import cv2

# VGG 自带的一个常量，之前VGG训练通过归一化，所以现在同样需要作此操作
VGG_MEAN = [103.939, 116.779, 123.68]  # rgb 三通道的均值


class VGGNet():
    '''
    创建 vgg16 网络 结构
    从模型中载入参数
    '''

    def __init__(self, data_dict):
        '''
        传入vgg16模型
        :param data_dict: vgg16.npy (字典类型)
        '''
        self.data_dict = data_dict

    def get_conv_filter(self, name):
        '''
        得到对应名称的卷积层
        :param name: 卷积层名称
        :return: 该卷积层输出
        '''
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        '''
        获得名字为name的全连接层权重
        :param name: 连接层名称
        :return: 该层权重
        '''
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        '''
        获得名字为name的全连接层偏置
        :param name: 连接层名称
        :return: 该层偏置
        '''
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        '''
        创建一个卷积层
        :param x:
        :param name:
        :return:
        '''
        # 在写计算图模型的时候，加一些必要的 name_scope，这是一个比较好的编程规范
        # 可以防止命名冲突， 二可视化计算图的时候比较清楚
        with tf.name_scope(name):
            # 获得 w 和 b
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)

            # 进行卷积计算
            h = tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding='SAME')
            '''
            因为此刻的 w 和 b 是从外部传递进来，所以使用 tf.nn.conv2d()
            tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu = None, name = None) 参数说明：
            input 输入的tensor， 格式[batch, height, width, channel]
            filter 卷积核 [filter_height, filter_width, in_channels, out_channels] 
                分别是：卷积核高，卷积核宽，输入通道数，输出通道数
            strides 步长 卷积时在图像每一维度的步长，长度为4
            padding 参数可选择 “SAME” “VALID”

            '''
            # 加上偏置
            h = tf.nn.bias_add(h, conv_b)
            # 使用激活函数
            h = tf.nn.relu(h)
            return h

    def pooling_layer(self, x, name):
        '''
        创建池化层
        :param x: 输入的tensor
        :param name: 池化层名称
        :return: tensor
        '''
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],  # 核参数， 注意：都是4维
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name
                              )

    def fc_layer(self, x, name, activation=tf.nn.relu):
        '''
        创建全连接层
        :param x: 输入tensor
        :param name: 全连接层名称
        :param activation: 激活函数名称
        :return: 输出tensor
        '''
        with tf.name_scope(name, activation):
            # 获取全连接层的 w 和 b
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            # 矩阵相乘 计算
            h = tf.matmul(x, fc_w)
            # 　添加偏置
            h = tf.nn.bias_add(h, fc_b)
            # 因为最后一层是没有激活函数relu的，所以在此要做出判断
            if activation is None:
                return h
            else:
                return activation(h)

    def flatten_layer(self, x, name):
        '''
        展平
        :param x: input_tensor
        :param name:
        :return: 二维矩阵
        '''
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            # 计算后三维合并后的大小
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            # 形成一个二维矩阵
            x = tf.reshape(x, [-1, dim])
            return x

    def build(self, x_rgb):
        '''
        创建vgg16 网络
        :param x_rgb: [1, 224, 224, 3]
        :return:
        '''
        start_time = time.time()
        # print('模型开始创建……')
        # 将输入图像进行处理，将每个通道减去均值
        # r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        '''
        tf.split(value, num_or_size_split, axis=0)用法：
        value:输入的Tensor
        num_or_size_split:有两种用法：
            1.直接传入一个整数，代表会被切成几个张量，切割的维度有axis指定
            2.传入一个向量，向量长度就是被切的份数。传入向量的好处在于，可以指定每一份有多少元素
        axis, 指定从哪一个维度切割
        因此，上一句的意思就是从第4维切分，分为3份，每一份只有1个元素
        '''
        # 将 处理后的通道再次合并起来
        # x_bgr = tf.concat([b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

        #        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 开始构建卷积层
        # vgg16 的网络结构
        # 第一层：2个卷积层 1个pooling层
        # 第二层：2个卷积层 1个pooling层
        # 第三层：3个卷积层 1个pooling层
        # 第四层：3个卷积层 1个pooling层
        # 第五层：3个卷积层 1个pooling层
        # 第六层： 全连接
        # 第七层： 全连接
        # 第八层： 全连接

        # 这些变量名称不能乱取，必须要和vgg16模型保持一致,
        # 另外，将这些卷积层用self.的形式，方便以后取用方便
        self.conv1_1 = self.conv_layer(x_rgb, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        # print('创建模型结束：%4ds' % (time.time() - start_time))



def get_row_col(num_pic):
    '''
    计算行列的值
    :param num_pic: 特征图的数量
    :return:
    '''
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

# def visualize_feature_map(feature_batch):
#     '''
#     创建特征子图，创建叠加后的特征图
#     :param feature_batch: 一个卷积层所有特征图
#     :return:
#     '''
#     feature_map = np.squeeze(feature_batch, axis=0)
#
#     feature_map_combination = []
#     plt.figure(figsize=(8, 7))
#
#     # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
#     #num_pic = feature_map.shape[2]
#
#     row, col = get_row_col(25)
#     # 将 每一层卷积的特征图，拼接层 5 × 5
#     for i in range(0, 25):
#         feature_map_split = feature_map[:, :, i]
#         feature_map_combination.append(feature_map_split)
#         plt.subplot(row, col, i+1)
#         plt.imshow(feature_map_split)
#         plt.axis('off')
#
#     #plt.savefig('./mao_feature/feature_map2.png') # 保存图像到本地
#     plt.show()


def visualize_feature_map_sum(feature_batch):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    feature_map = np.squeeze(feature_batch, axis=0)

    feature_map_combination = []

    # 取出 featurn map 的数量
    num_pic = feature_map.shape[2]

    # 将 每一层卷积的特征图，拼接层 5 × 5
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)

    # 按照特征图 进行 叠加代码

    feature_map_sum = sum(one for one in feature_map_combination)
    return feature_map_sum
    # plt.imshow(feature_map_sum)
    # #plt.savefig('./mao_feature/feature_map_sum2.png') # 保存图像到本地
    # plt.show()


def get_feature(image, layer):
    # content = tf.placeholder(tf.float32, shape=[1, 450, 620, 3])
    vgg16_npy_pyth = 'vgg16.npy'
    # content = tf.concat([image, image, image], axis=-1)
    content = image
    # 载入模型， 注意：在python3中，需要添加一句： encoding='latin1'
    data_dict = np.load(vgg16_npy_pyth, encoding='latin1', allow_pickle=True).item()

    # 创建图像的 vgg 对象
    vgg_for_content = VGGNet(data_dict)

    # 创建 每个 神经网络
    vgg_for_content.build(content)

    content_features = [vgg_for_content.conv1_2,
                        vgg_for_content.conv2_2,
                        vgg_for_content.conv3_3,
                        vgg_for_content.conv4_3,
                        vgg_for_content.conv5_3,
                        ]

    # init_op = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     content_features = sess.run([content_features],
    #                                 feed_dict={
    #                                     content: image
    #                                 })

    conv1 = content_features[0]
    conv2 = content_features[1]
    conv3 = content_features[2]
    conv4 = content_features[3]
    conv5 = content_features[4]
    if layer == 'conv1':
        return conv1
    if layer == 'conv2':
        return conv2
    if layer == 'conv3':
        return conv3
    if layer == 'conv4':
        return conv4
    if layer == 'conv5':
        return conv5
    # # 查看 每个 特征 子图
    # visualize_feature_map(conv3)
    #
    # # 查看 叠加后的 特征图
    # visualize_feature_map_sum(conv3)

def Perceptual_Loss(Orignal_image, Generate_image):
    orignal_conv3_feature = get_feature(Orignal_image, 'conv3')
    orignal_conv4_feature = get_feature(Orignal_image, 'conv4')
    Generate_conv3_feature = get_feature(Generate_image, 'conv3')
    Generate_conv4_feature = get_feature(Generate_image, 'conv4')
    conv3_LOSS = tf.reduce_mean(tf.abs(orignal_conv3_feature - Generate_conv3_feature))
    conv4_LOSS = tf.reduce_mean(tf.abs(orignal_conv4_feature - Generate_conv4_feature))
    Perceptualloss = conv3_LOSS + conv4_LOSS
    return Perceptualloss


def TV_LOSS(batchimg):
    TV_norm = tf.reduce_sum(tf.abs(batchimg), axis=[1, 2, 3])
    E = tf.reduce_mean(TV_norm)
    return E








