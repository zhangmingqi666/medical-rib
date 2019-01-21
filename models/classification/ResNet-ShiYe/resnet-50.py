import tensorflow as tf
from tensorflow.python.training import moving_averages
from ResNet.load_data import load_tfrecord

fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d

TRAINING_STEPS = 10000
BATCH_SIZE = 16
OUTPUT_NODE = 2
CHANNEL = 1

# OUTPUT_NODE = 17
# CHANNEL = 3


# create weight variable
def create_var(name, shape, initializer, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable)


# conv2d layer
def conv2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        kernel = create_var("kernel", [kernel_size, kernel_size, num_inputs, num_outputs], conv2d_initializer())
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")


# fully connected layer
def fc(x, num_outputs, scope="fc"):
    num_inputs = x.get_shape()[-1]
    with tf.variable_scope(scope):
        weight = create_var("weight", [num_inputs, num_outputs], fc_initializer())
        bias = create_var("bias", [num_outputs, ], tf.zeros_initializer())
        return tf.nn.xw_plus_b(x, weight, bias)


# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = create_var("beta", [num_inputs, ], initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs, ], initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_var("moving_mean", [num_inputs, ], initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_var("moving_variance", [num_inputs], initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


# avg pool layer
def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding="VALID")


# max pool layer
def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1], [1, stride, stride, 1], padding="SAME")


class ResNet50(object):
    def __init__(self, inputs, num_classes=OUTPUT_NODE, is_training=True,
                 scope="resnet50"):
        self.inputs = inputs
        self.is_training = is_training
        self.num_classes = num_classes

        with tf.variable_scope(scope):
            # construct the model
            net = conv2d(inputs, 64, 7, 2, scope="conv1") # -> [batch, 112, 112, 64]
            net = tf.nn.relu(batch_norm(net, is_training=self.is_training, scope="bn1"))
            net = max_pool(net, 3, 2, scope="maxpool1")  # -> [batch, 56, 56, 64]
            net = self._block(net, 256, 3, init_stride=1, is_training=self.is_training,  scope="block2")           # -> [batch, 56, 56, 256]
            net = self._block(net, 512, 4, is_training=self.is_training, scope="block3")  # -> [batch, 28, 28, 512]
            net = self._block(net, 1024, 6, is_training=self.is_training, scope="block4")  # -> [batch, 14, 14, 1024]
            net = self._block(net, 2048, 3, is_training=self.is_training, scope="block5")   # -> [batch, 7, 7, 2048]
            net = avg_pool(net, 7, scope="avgpool5")    # -> [batch, 1, 1, 2048]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # -> [batch, 2048]
            self.logits = fc(net, self.num_classes, "fc6")       # -> [batch, num_classes]
            self.prediction = tf.nn.softmax(self.logits)

    def _block(self, x, n_out, n, init_stride=2, is_training=True, scope="block"):
        with tf.variable_scope(scope):
            h_out = n_out // 4
            out = self._bottleneck(x, h_out, n_out, stride=init_stride, is_training=is_training, scope="bottlencek1")
            for i in range(1, n):
                out = self._bottleneck(out, h_out, n_out, is_training=is_training, scope=("bottlencek%s" % (i + 1)))
            return out

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):
        """ A residual bottleneck unit"""
        n_in = x.get_shape()[-1]
        if stride is None:
            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):
            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")
            h = batch_norm(h, is_training=is_training, scope="bn_1")
            h = tf.nn.relu(h)
            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")
            h = batch_norm(h, is_training=is_training, scope="bn_2")
            h = tf.nn.relu(h)
            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")
            h = batch_norm(h, is_training=is_training, scope="bn_3")

            if n_in != n_out:
                shortcut = conv2d(x, n_out, 1, stride=stride, scope="conv_4")
                shortcut = batch_norm(shortcut, is_training=is_training, scope="bn_4")
            else:
                shortcut = x
            return tf.nn.relu(shortcut + h)


if __name__ == "__main__":

    x = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_NODE], name='y-input')
    global_step = tf.Variable(0, trainable=False)
    resnet50 = ResNet50(x)
    y = resnet50.logits
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean # + ALPHA * tf.add_n(tf.get_collection('loss'))
    learning_rate = 0.001 # tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, NUM_EXAMPLE / BATCH_SIZE, LEARNING_RATE_DELAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    print(resnet50.logits)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        img, label = load_tfrecord(tf_records_file_name='train.bone_tfrecords', channel=CHANNEL)
        img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=BATCH_SIZE, capacity=2000, min_after_dequeue=1000)
        labels = tf.one_hot(label_batch, OUTPUT_NODE, 1, 0)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(TRAINING_STEPS):
            batch_xs, batch_ys = sess.run([img_batch, labels])
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: batch_xs, y_: batch_ys})

            # print loss value per 10 steps
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

        coord.request_stop()
        coord.join()
