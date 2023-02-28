import tensorflow as tf
import numpy as np
import integrated_gradients_tf as ig

class BatchFeeder:
    """ Simple iterator for feeding a subset of numpy matrix into tf network.
    validation data has same size of mini batch
     Parameter
    ----------------
    X: ndarray
    y: ndarray
    batch_size: mini batch size
    """

    def __init__(self, x_, y_, batch_size, valid=False, ini_random=True):
        """check whether X and Y have the matching sample size."""
        assert len(x_) == len(y_)
        self.n = len(x_)
        self.X = x_
        self.y = y_
        self.index = 0
        # self.base_index = np.arange(len(X))
        if ini_random:
            _ = self.randomize(np.arange(len(x_)))
        if valid:
            self.create_validation(batch_size)
        self.batch_size = batch_size
        self.base_index = np.arange(self.n)
        self.val = None

    def create_validation(self, batch_size):
        self.val = (self.X[-1*int(batch_size):], self.y[-1*int(batch_size):])
        self.X = self.X[:-1*int(batch_size)]
        self.y = self.y[:-1*int(batch_size)]
        self.n = len(self.X)-int(batch_size)

    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.base_index = self.randomize(self.base_index)
        ret_x = self.X[self.index:self.index+self.batch_size]
        ret_y = self.y[self.index:self.index+self.batch_size]
        self.index += self.batch_size
        return ret_x, ret_y

    def randomize(self, index):
        np.random.shuffle(index)
        self.y = self.y[index]
        self.X = self.X[index]
        return index

class testnet:
    def __init__(self):
        # Reset all existing tensors
        self.dimensions = [128, 64]
        tf.reset_default_graph()
        self.built = False
        self.sesh = tf.Session()
        self.e = 0
        self.ops = self.build()
        self.sesh.run(tf.global_variables_initializer())
    
    def build(self):
        # Placeholders for input and dropout probs.
        if self.built:
            return -1
        else:
            self.built = True
            
        x = tf.placeholder(tf.float32, shape=[None, 28, 28], name="x")
        _x = tf.contrib.slim.flatten(x)
        y = tf.placeholder(tf.int64, shape=[None, 10], name="y")
        
        # Buildin IG model
        inter, stepsize, ref = ig.linear_inpterpolation(_x, num_steps=50)
        
        # Fully connected encoder.
        with tf.variable_scope("predictor"):
            dense = _x
            for dim in self.dimensions:
                dense = tf.contrib.slim.fully_connected(dense, dim, activation_fn=tf.nn.relu)
            dense = tf.contrib.slim.fully_connected(dense, 10, activation_fn=tf.identity)
            prediction = tf.nn.softmax(dense)
        
        with tf.variable_scope("predictor", reuse=True):
            dense2 = inter
            for dim in self.dimensions:
                dense2 = tf.contrib.slim.fully_connected(dense2, dim, activation_fn=tf.nn.relu)
            dense2 = tf.contrib.slim.fully_connected(dense2, 10, activation_fn=tf.identity)
            prediction2 = tf.nn.softmax(dense2)
        
        explanations = []
        for i in range(10):
            explanations.append(ig.build_ig(inter, stepsize, prediction2[:, i], num_steps=50))
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=y))                                   
                                            
        # Define cost as the sum of KL and reconstrunction ross with BinaryXent.
        with tf.name_scope("cost"):
            # average over minibatch
            cost = loss
        
        # Defining optimization procedure.
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer()
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
            train = optimizer.apply_gradients(clipped, name="minimize_cost")
            
        # Exporting out the operaions as dictionary
        return dict(
            x = x,
            y = y,
            prediction = prediction,
            cost = cost,
            train = train,
            explanations = explanations
        )
    
    # Closing session
    def close(self):
        self.sesh.close()
    
    # training procedure.
    def train(self, X, epochs, valid=None):
        # Making the saver object.
        saver = tf.train.Saver()
        
        # Defining the number of batches per epoch
        batch_num = int(np.ceil(X.n*1.0/X.batch_size))
        
        e = 0
        while e < epochs:
            
            for i in range(batch_num):
                #Training happens here.
                batch = X.next()
                feed_dict = {self.ops["x"]: batch[0], self.ops["y"]: batch[1]}
                ops_to_run = [self.ops["prediction"],\
                              self.ops["cost"],\
                              self.ops["train"]]
                prediction, cost, _ = self.sesh.run(ops_to_run, feed_dict)
            
            self.e+=1
            e+= 1
                
            print "Epoch:"+str(self.e)
    
    # Encode examples
    def predict(self, x):
        feed_dict = {self.ops["x"]: x}
        return self.sesh.run(self.ops["prediction"], feed_dict=feed_dict)
    
    def explain(self, x):
        feed_dict = {self.ops["x"]: x}
        return self.sesh.run(self.ops["explanations"], feed_dict=feed_dict)