import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1

# VIS discriminator
class Discriminator1(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.kernel_conv1,self.bias_conv1=self._create_variables(1, 16, 3, scope='conv1')
			self.kernel_conv2, self.bias_conv2 = self._create_variables(16, 32, 3, scope='conv2')
			self.kernel_conv3, self.bias_conv3 = self._create_variables(32, 64, 3, scope='conv3')
			self.kernel_conv4, self.bias_conv4 = self._create_variables(64, 128, 3, scope='conv4')
			# kernel5 only use in patch_dis and pix_dis
			self.kernel_conv5, self.bias_conv5 = self._create_variables(128, 1, 3, scope='conv5')

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.get_variable('kernel', shape=shape,
									 initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias =tf.get_variable('bias',shape=[output_filters],initializer=tf.zeros_initializer())
		return (kernel, bias)


	def discrim(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img

		# control stride (B 256 256 1)>(B 128 128 16)>(B 64 64 32)>(B 32 32 64)>(B 16 16 128)
		out = conv2d_1(out, self.kernel_conv1, self.bias_conv1, [1, 2, 2, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv2, self.bias_conv2, [1, 2, 2, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn2' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv3, self.bias_conv3, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv4, self.bias_conv4, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4' , Reuse=reuse)

		# (B 16 16 128)>(B 1 1 128)>(B 128)
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				shape=out.get_shape().as_list()
				out = tf.layers.average_pooling2d(inputs=out, pool_size=shape[1], strides=1, padding='VALID',
									  name='global_average_pool')
				out_flatten=tf.reshape(out,shape=[-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])

		# fc(B 128) >（B,1）
		with tf.variable_scope(self.scope):
			with tf.variable_scope('fc'):
				out_logit = tf.layers.dense(out_flatten, 1, activation=None, use_bias=True, trainable=True,reuse=reuse)
		# activate to (0-1)
		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

	def discrim_patch(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		# control stride (B 256 256 1)>(B 128 128 16)>(B 64 64 32)>(B 32 32 64)>(B 16 16 128)
		out = conv2d_1(out, self.kernel_conv1, self.bias_conv1, [1, 2, 2, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv2, self.bias_conv2, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv3, self.bias_conv3, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv4, self.bias_conv4, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4' , Reuse=reuse)

		# (B 16 16 128)>(B 16 16 1)
		# you need to add the kernel conv5 (128, 1, 3)
		out_logit = conv2d_1(out, self.kernel_conv5, self.bias_conv5, [1, 1, 1, 1], use_relu=False, use_BN=True,
					   Scope=self.scope + '/bn5' , Reuse=reuse)
		# receptive filed: 63
		# activate to (0-1)
		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

	def discrim_pixel(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		# control stride (B 256 256 1)>(B 256 256 16)>(B 256 256 32)>(B 256 256 64)>(B 256 256 128)
		out = conv2d_1(out, self.kernel_conv1, self.bias_conv1, [1, 1, 1, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv2, self.bias_conv2, [1, 1, 1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv3, self.bias_conv3, [1, 1, 1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3' , Reuse=reuse)
		out = conv2d_1(out, self.kernel_conv4, self.bias_conv4, [1, 1, 1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4' , Reuse=reuse)

		# (B 256 256 128)>(B 256 256 1)
		# you need to add the kernel conv5 whith shape of # you need to add the kernel conv5 (128, 1, 3)
		out_logit = conv2d_1(out, self.kernel_conv5, self.bias_conv5, [1, 2, 2, 1], use_relu=False, use_BN=False,
					   Scope=self.scope + '/bn5' , Reuse=reuse)

		# activate to (0-1)
		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

def conv2d_1(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, Reuse = None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out


# IR discriminator
class Discriminator2(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.kernel_conv1,self.bias_conv1=self._create_variables(1, 16, 3, scope='conv1')
			self.kernel_conv2, self.bias_conv2 = self._create_variables(16, 32, 3, scope='conv2')
			self.kernel_conv3, self.bias_conv3 = self._create_variables(32, 64, 3, scope='conv3')
			self.kernel_conv4, self.bias_conv4 = self._create_variables(64, 128, 3, scope='conv4')
			# only use in dis_patch and dis_pixel
			self.kernel_conv5, self.bias_conv5 = self._create_variables(128, 1, 3, scope='conv5')


	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.variable_scope(scope):
			kernel = tf.get_variable('kernel', shape=shape,
									 initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
			bias =tf.get_variable('bias',shape=[output_filters],initializer=tf.zeros_initializer())
		return (kernel, bias)

	def discrim(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		# control stride (B 256 256 1)>(B 128 128 16)>(B 64 64 32)>(B 32 32 64)>(B 16 16 128)
		out = conv2d_2(out, self.kernel_conv1, self.bias_conv1, [1, 2, 2, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1' , Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv2, self.bias_conv2, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2' , Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv3, self.bias_conv3, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3' , Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv4, self.bias_conv4, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4' , Reuse=reuse)

		# (B 16 16 128)>(B 1 1 128)>(B 128)
		with tf.variable_scope(self.scope):
			with tf.variable_scope('flatten1'):
				shape=out.get_shape().as_list()
				out = tf.layers.average_pooling2d(inputs=out, pool_size=shape[1], strides=1, padding='VALID',
									  name='global_vaerage_pool')
				out_flatten=tf.reshape(out,shape=[-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])
		# fc (B,128)>（B,1）
		with tf.variable_scope(self.scope):
			with tf.variable_scope('fc'):
				out_logit = tf.layers.dense(out_flatten, 1, activation=None, use_bias=True, trainable=True,reuse=reuse)
		# activate to (0-1)
		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

	def discrim_patch(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		# control stride (B 256 256 1)>(B 128 128 16)>(B 64 64 32)>(B 32 32 64)>(B 16 16 128)
		out = conv2d_2(out, self.kernel_conv1, self.bias_conv1, [1, 2, 2, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv2, self.bias_conv2, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv3, self.bias_conv3, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv4, self.bias_conv4, [1, 2, 2, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4', Reuse=reuse)

		# (B 16 16 128)>(B 16 16 1)
		# you need to add the kernel conv5 (128, 1, 3)
		out_logit = conv2d_2(out, self.kernel_conv5, self.bias_conv5, [1,1, 1, 1], use_relu=False, use_BN=True,
					   Scope=self.scope + '/bn5' , Reuse=reuse)
		# receptive field:63

		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

	def discrim_pixel(self, img, reuse):
		if len(img.shape) != 4:
			img = tf.expand_dims(img, -1)
		out = img
		# control stride (B 256 256 1)>(B 256 256 16)>(B 256 256 32)>(B 256 256 64)>(B 256 256 128)
		out = conv2d_2(out, self.kernel_conv1, self.bias_conv1, [1, 1,1, 1], use_relu=True, use_BN=False,
					   Scope=self.scope + '/bn1', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv2, self.bias_conv2, [1, 1,1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn2', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv3, self.bias_conv3, [1, 1,1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn3', Reuse=reuse)
		out = conv2d_2(out, self.kernel_conv4, self.bias_conv4, [1, 1,1, 1], use_relu=True, use_BN=True,
					   Scope=self.scope + '/bn4', Reuse=reuse)

		# (B 256 256 128)>(B 256 256 1)
		# you need to add the kernel conv5 whith shape of # you need to add the kernel conv5 (128, 1, 3)
		out_logit = conv2d_2(out, self.kernel_conv5, self.bias_conv5, [1, 1, 1, 1], use_relu=False, use_BN=False,
					   Scope=self.scope + '/bn5' , Reuse=reuse)

		# activate to (0-1)
		out_activate=tf.nn.tanh(out_logit)/2+0.5
		return out_activate, out_logit

def conv2d_2(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, Reuse = None):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(x_padded, kernel, strides, padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if use_BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
	if use_relu:
		out = tf.nn.relu(out)
	return out






