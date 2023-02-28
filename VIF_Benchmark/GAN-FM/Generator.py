import tensorflow as tf


WEIGHT_INIT_STDDEV = 0.05

def conv_block(x,kernel,use_relu=True,Scope=None,BN=None):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training=True)
    if use_relu:
        out = tf.nn.relu(out)
    return out

def UnetConv2(x,kernel1,kernel2,use_relu=True,Scope=None,BN=False):
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(x_padded, kernel1, strides=[1, 1, 1, 1], padding='VALID')
    if BN:
        with tf.variable_scope(Scope+'/bn1'):
            out = tf.layers.batch_normalization(out, training=True)
    if use_relu:
        out = tf.nn.relu(out)

    out_padded = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    out = tf.nn.conv2d(out_padded, kernel2, strides=[1, 1, 1, 1], padding='VALID')
    if BN:
        with tf.variable_scope(Scope+'/bn2'):
            out = tf.layers.batch_normalization(out, training=True)
    if use_relu:
        out = tf.nn.relu(out)
    return out

def upsample(inputs, shape):
    out = tf.image.resize_bilinear(images=inputs,size=[shape[1],shape[2]])
    return out

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

class Generator(object):

    def __init__(self,scope_name):
        self.scope=scope_name
        with tf.variable_scope(self.scope):
            self.first_kernel = self._create_variables(2, 16, 3, scope='first1')

            self.en1_kernel1= self._create_variables(16, 32, 3, scope='en1_conv1')
            self.en1_kernel2= self._create_variables(32, 32, 3, scope='en1_conv2')
            self.en2_kernel1= self._create_variables(32, 64, 3, scope='en2_conv1')
            self.en2_kernel2= self._create_variables(64, 64, 3, scope='en2_conv2')
            self.en3_kernel1= self._create_variables(64, 128, 3, scope='en3_conv1')
            self.en3_kernel2= self._create_variables(128, 128, 3, scope='en3_conv2')
            self.en4_kernel1= self._create_variables(128, 256, 3, scope='en4_conv1')
            self.en4_kernel2= self._create_variables(256,256, 3, scope='en4_conv2')

            self.h1_PT_hd3_kernel=self._create_variables(32,32,3, scope='h1_to_dh3_conv')
            self.h2_PT_hd3_kernel=self._create_variables(64,32,3, scope='h2_to_dh3_conv')
            self.h3_cat_hd3_kernel=self._create_variables(128,32,3, scope='h3_cat_hd3_conv')
            self.hd4_UT_hd3_kernel=self._create_variables(256,32,3, scope='hd4_UT_hd3_conv')

            self.hd3_kernel=self._create_variables(128,128,3, scope='hd3_conv')

            self.h1_PT_hd2_kernel=self._create_variables(32,32,3, scope='h1_to_dh2_conv')
            self.h2_cat_hd2_kernel = self._create_variables(64, 32, 3, scope='h2_cat_hd2_conv')
            self.hd3_UT_hd2_kernel = self._create_variables(128, 32, 3, scope='hd3_UT_hd2_conv')
            self.hd4_UT_hd2_kernel=self._create_variables(256, 32, 3, scope='hd4_UT_hd2_conv')

            self.hd2_kernel = self._create_variables(128, 128, 3, scope='hd2_conv')

            self.h1_cat_hd1_kernel = self._create_variables(32, 32, 3, scope='h1_cat_hd1_conv')
            self.hd2_UT_hd1_kernel=self._create_variables(128, 32, 3, scope='hd2_UT_hd1_conv')
            self.hd3_UT_hd1_kernel=self._create_variables(128, 32, 3, scope='hd3_UT_hd1_conv')
            self.hd4_UT_hd1_kernel = self._create_variables(256,32, 3, scope='hd4_UT_hd1_conv')

            self.hd1_kernel = self._create_variables(128, 128, 3, scope='hd1_conv')

            self.final_kernel1=self._create_variables(128, 64, 3, scope='final_out_conv1')
            self.final_kernel2 = self._create_variables(64, 1, 3, scope='final_out_conv2')


    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable('kernel', shape=shape,
                                     initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
        return kernel

    def transform(self,vis, ir):
        img = tf.concat([vis, ir], 3)
        inputs=img

        inputs = conv_block(inputs,kernel=self.first_kernel,use_relu=True,Scope=self.scope+'first_conv',BN=False)

        #encoder1
        h1=UnetConv2(inputs,kernel1=self.en1_kernel1,kernel2=self.en1_kernel2,use_relu=True,Scope=self.scope+ '/encoder1/b',BN=False)
        #shape(none,256,256,32)

        # maxpool
        h2=tf.nn.max_pool(h1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        #encoder2
        h2=UnetConv2(h2,kernel1=self.en2_kernel1,kernel2=self.en2_kernel2,use_relu=True,Scope=self.scope+ '/encoder2/b',BN=False)
        #shape(128 128 64)

        h3=tf.nn.max_pool(h2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        #encoder3
        h3=UnetConv2(h3,kernel1=self.en3_kernel1,kernel2=self.en3_kernel2,use_relu=True,Scope=self.scope+ '/encoder3/b' ,BN=False)
        #shape(64 64 128)

        h4=tf.nn.max_pool(h3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #shape(32 32 256)
        #encoder4
        hd4=UnetConv2(h4,kernel1=self.en4_kernel1,kernel2=self.en4_kernel2,use_relu=True,Scope=self.scope+ '/encoder4/b' ,BN=False)
        #shape(none,32,32,256)


        # stage 3
        h1_PT_hd3=tf.nn.max_pool(h1,ksize=[1,2,2,1],strides=[1,4,4,1],padding='SAME')
        h1_PT_hd3=conv_block(h1_PT_hd3,kernel=self.h1_PT_hd3_kernel,use_relu=True,Scope=self.scope+'/h1_to_dh3',BN=False)
        # (none,256,256,32)>(none 64 64 32)
        h2_PT_hd3 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h2_PT_hd3=conv_block(h2_PT_hd3,kernel=self.h2_PT_hd3_kernel,use_relu=True,Scope=self.scope+'/h2_to_dh3',BN=False)
        h3_cat_hd3=conv_block(h3,kernel=self.h3_cat_hd3_kernel,use_relu=True,Scope=self.scope+'/h3_cat_hd3',BN=False)

        hd4_UT_hd3=upsample(hd4,shape=tf.shape(h3))
        hd4_UT_hd3=conv_block(hd4_UT_hd3,kernel=self.hd4_UT_hd3_kernel,use_relu=True,Scope=self.scope+'/hd4_ut_hd3',BN=False)


        hd3= tf.concat([h1_PT_hd3,h2_PT_hd3,h3_cat_hd3,hd4_UT_hd3],3)
        #shape(none 64 64 32*4)
        hd3 = conv_block(hd3, kernel=self.hd3_kernel, use_relu=True, Scope=self.scope + '/hd3', BN=False)


        #stage 2
        h1_PT_hd2=tf.nn.max_pool(h1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h1_PT_hd2=conv_block(h1_PT_hd2,kernel=self.h1_PT_hd2_kernel,use_relu=True,Scope=self.scope+'/h1_to_hd2',BN=False)#shape(none 64 64 32)

        h2_cat_hd2 = conv_block(h2, kernel=self.h2_cat_hd2_kernel , use_relu=True, Scope=self.scope + '/h2_cat_hd2', BN=False)


        hd3_UT_hd2 = upsample(hd3, shape=tf.shape(h2))
        hd3_UT_hd2 = conv_block(hd3_UT_hd2, kernel=self.hd3_UT_hd2_kernel, use_relu=True, Scope=self.scope + '/hd3_ut_hd2', BN=False)


        hd4_UT_hd2 = upsample(hd4, shape=tf.shape(h2))
        hd4_UT_hd2=conv_block(hd4_UT_hd2, kernel= self.hd4_UT_hd2_kernel, use_relu=True, Scope=self.scope + '/hd4_ut_hd2', BN=False)


        hd2=tf.concat([h1_PT_hd2,h2_cat_hd2,hd3_UT_hd2,hd4_UT_hd2],3)
        #shape(none 128 128 4*32)
        hd2=conv_block(hd2,kernel=self.hd2_kernel,use_relu=True,Scope=self.scope+'/hd2',BN=False)


        #stage 1
        h1_cat_hd1=conv_block(h1, kernel= self.h1_cat_hd1_kernel , use_relu=True, Scope=self.scope + '/h1_cat_hd1', BN=False)

        hd2_UT_hd1=upsample(hd2, shape=tf.shape(h1))
        hd2_UT_hd1=conv_block(hd2_UT_hd1, kernel= self.hd2_UT_hd1_kernel, use_relu=True, Scope=self.scope + '/hd2_ut_hd1', BN=False)

        hd3_UT_hd1=upsample(hd3, shape=tf.shape(h1))
        hd3_UT_hd1=conv_block(hd3_UT_hd1, kernel= self.hd3_UT_hd1_kernel, use_relu=True, Scope=self.scope + '/hd3_ut_hd1', BN=False)

        hd4_UT_hd1=upsample(hd4, shape=tf.shape(h1))
        hd4_UT_hd1=conv_block(hd4_UT_hd1, kernel= self.hd4_UT_hd1_kernel, use_relu=True, Scope=self.scope + '/hd4_ut_hd1', BN=False)

        hd1=tf.concat([h1_cat_hd1,hd2_UT_hd1,hd3_UT_hd1,hd4_UT_hd1],3)
        hd1 = conv_block(hd1, kernel=self.hd1_kernel, use_relu=True, Scope=self.scope + '/hd1', BN=False)
        # shape(256 256 4*32)

        unet_out=hd1

        out=instance_norm(conv_block(unet_out, kernel= self.final_kernel1, use_relu=False, Scope=self.scope + '/final1', BN=False,
                            ),name=self.scope+'/final1_bn1')

        out=conv_block(out, kernel= self.final_kernel2, use_relu=False, Scope=self.scope + '/final2', BN=False,
                              )

        out = tf.nn.tanh(out) / 2 + 0.5
        return out











