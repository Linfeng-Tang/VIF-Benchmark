import tensorflow as tf
from utils import weights_spectral_norm
from ops import *
from utils import *

class Illumination_classifier():
    def illumination_classifier(self, input_image, reuse=False):
        # features = [batch_size, 256, 256, 128]
        channel = 16
        with tf.compat.v1.variable_scope('Classifier', reuse=reuse):
            x = input_image
            for i in range(4):
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv' + str(i + 1), sn=False, norm=False)
                x = lrelu(x)
                channel = channel * 2
            x = tf.reduce_mean(x, axis=[1, 2])
            x = tf.layers.flatten(x)
            x = tf.layers.dense(inputs=x, units=128)
            out = tf.layers.dense(inputs=x, units=2)
            out = tf.abs(out)
        return out
    
class PIAFusion():
    def CMDAF(self, F_vi, F_ir):
        sub_vi_ir = tf.subtract(F_vi, F_ir)
        sub_w_vi_ir = tf.reduce_mean(sub_vi_ir, axis=[1, 2],  keepdims=True)  # Global Average Pooling
        w_vi_ir = tf.nn.sigmoid(sub_w_vi_ir)

        sub_ir_vi = tf.subtract(F_ir, F_vi)
        sub_w_ir_vi = tf.reduce_mean(sub_ir_vi, axis=[1, 2],  keepdims=True)  # Global Average Pooling
        w_ir_vi = tf.nn.sigmoid(sub_w_ir_vi)

        F_dvi = tf.multiply(w_vi_ir, sub_ir_vi) # 放大差分信号，此处是否应该调整为sub_ir_vi
        F_dir = tf.multiply(w_ir_vi, sub_vi_ir)

        F_fvi = tf.add(F_vi, F_dir)
        F_fir = tf.add(F_ir, F_dvi)
        return F_fvi, F_fir

    def Encoder(self, vi_image, ir_image, reuse=False):
        channel = 16
        with tf.compat.v1.variable_scope('encoder', reuse=reuse):
            x_ir = conv(ir_image, channel, kernel=1, stride=1, pad=0, pad_type='reflect', scope='conv5x5_ir')
            x_ir = lrelu(x_ir)
            x_vi = conv(vi_image, channel, kernel=1, stride=1, pad=0, pad_type='reflect', scope='conv5x5_vi')
            x_vi = lrelu(x_vi)
            block_num = 4
            for i in range(block_num):  # the number of resblocks in feature extractor is 3
                input_ir = x_ir
                input_vi = x_vi
                with tf.compat.v1.variable_scope('Conv{}'.format(i + 1), reuse=False):
                        # conv1
                    x_ir = conv(input_ir, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv3x3')
                    x_ir = lrelu(x_ir)
                with tf.compat.v1.variable_scope('Conv{}'.format(i + 1), reuse=True):
                    # conv1
                    x_vi = conv(input_vi, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv3x3')
                    x_vi = lrelu(x_vi)
                # # want to use one convolutional layer to extract features with consistent distribution from various sourece images  
                if i != block_num - 1:
                    channel = channel * 2
                    x_vi, x_ir = self.CMDAF(x_vi, x_ir)
            print('channel:',  channel)
            return x_vi, x_ir


    def Decoder(self, x, reuse=False):
        channel = x.get_shape().as_list()[-1]
        print('channel:', channel)

        with tf.compat.v1.variable_scope('decoder', reuse=reuse):
            block_num = 4
            for i in range(block_num):  # the number of resblocks in feature extractor is 3

                features = x
                x = conv(features, channel, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv{}'.format(i + 1))
                x = lrelu(x)
                channel = channel / 2
            print('final channel:', channel)
            x = conv(x, 1, kernel=1, stride=1, pad=0, pad_type='reflect', scope='conv1x1')
            x = tf.nn.tanh(x) / 2 + 0.5
            return x

    def PIAFusion(self, vi_image, ir_image, reuse=False, Feature_out=True):
        vi_stream, ir_stream = self.Encoder(vi_image=vi_image, ir_image=ir_image, reuse=reuse)
        stream = tf.concat([vi_stream, ir_stream], axis=-1)
        fused_image = self.Decoder(stream, reuse=reuse)
        if Feature_out:
            return fused_image, vi_stream, ir_stream
        else:
            return fused_image
