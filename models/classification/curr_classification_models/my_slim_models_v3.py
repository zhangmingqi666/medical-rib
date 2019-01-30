import sys
import os
#sys.path.append("/Users/jiangyy/workspace/models/research/slim")
import tensorflow as tf
from tensorflow.python.training import moving_averages
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
BATCH_SIZE = 32
TRAIN_IMAGE_SIZE = 224
VAL_IMAGE_SIZE = 224
NUM_CLASSES = 2
L2_REGULARIZER = 0.001
LEARNING_RATE = 0.01
DISPLAY_STEP = 20



def get_dataset(mode='train', folder_dir='/home/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/trainSet', image_size=224):
    """
    获取数据集
    :param image_size: 输出图片的尺寸
    :param resize_image_max: 在切片前，将图片resize的最小尺寸
    :param resize_image_min: 在切片前，将图片resize的最大尺寸
    :param mode: 可以是 train val trainval 三者之一，对应于VOC数据集中的预先设定好的训练集、验证集
    :return: 返回元组，第一个参数是 tf.data.Dataset实例，第二个是数据集中元素数量
    """
    def get_image_paths_and_labels():
        # 从本地文件系统中，获取所有图片的绝对路径以及对应的标签
        if mode not in ['train', 'validation']:
            raise ValueError('Unknown mode: {}'.format(mode))
        CLASSES = ['unfragment', 'fragment']
        keys = []
        values = []
        for i, class_name in enumerate(CLASSES):
            subfolder = os.path.join(folder_dir, class_name)
            for f in os.listdir(subfolder):
                file = os.path.join(subfolder, f)
                keys.append(file)
                values.append(i)
        return keys, values

    def parse_image_by_path_fn(image_path):
        # 通过文件路径读取图片，并进行数据增强
        img_file = tf.read_file(image_path)
        cur_image = tf.image.decode_jpeg(img_file)
        return cur_image

    def preprocess_image(image):
        """Preprocesses the given image.

        Args:
          image: A `Tensor` representing an image of arbitrary size.
          output_height: The height of the image after preprocessing.
          output_width: The width of the image after preprocessing.
          is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.

        Returns:
          A preprocessed image.
        """
        image = tf.to_float(image)
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        image = tf.subtract(image, 128.0)
        image = tf.div(image, 128.0)
        return image

    paths, labels = get_image_paths_and_labels()

    # 建立tf.data.Dataset实例
    images_dataset = tf.data.Dataset.from_tensor_slices(paths).map(parse_image_by_path_fn).map(preprocess_image)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)#.map(lambda z:tf.one_hot(z, NUM_CLASSES))
    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=len(paths))
    return dataset.batch(batch_size=BATCH_SIZE), len(paths)


def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(L2_REGULARIZER)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, NUM_CLASSES, activation_fn=None, scope='fc8')
    return net


fc_initializer = tf.contrib.layers.xavier_initializer
conv2d_initializer = tf.contrib.layers.xavier_initializer_conv2d


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
    def __init__(self, inputs, num_classes=NUM_CLASSES, is_training=True,
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


def calc_mean_loss_with_sess(sess=None,):
    pass


def calc_mean_acc_with_sess(sess=None):
    pass


if __name__ == '__main__':

    train_folder = '/home/jiangyy/data_augmentation/trainSet'
    train_batch, train_set_size = get_dataset('train', folder_dir=train_folder, image_size=TRAIN_IMAGE_SIZE)
    train_iter = train_batch.make_initializable_iterator()
    train_next_element = train_iter.get_next()
    logger.info('train set created successfully with {} items.'.format(train_set_size))
    train_batch_num = (train_set_size - 1) // BATCH_SIZE + 1

    val_folder = '/home/jiangyy/data_augmentation/testSet'
    val_batch, val_set_size = get_dataset('validation', folder_dir=val_folder, image_size=VAL_IMAGE_SIZE)
    val_iter = val_batch.make_initializable_iterator()
    val_next_element = val_iter.get_next()
    logger.info('val set created successfully with {} items.'.format(val_set_size))
    val_batch_num = (val_set_size - 1) // BATCH_SIZE + 1

    max_iter = 1000
    X = tf.placeholder(tf.float32, [None, 224, 224, 1], name='x-input')
    Y = tf.placeholder(tf.float32, [None], name='y-input')

    #logits = vgg16(X)
    # vgg = nets.vgg
    # logits, _ = vgg.vgg_16(X, NUM_CLASSES)

    resnet50 = ResNet50(X)
    logits = resnet50.logits
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(Y, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=Y, predictions=tf.argmax(logits, axis=1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for iter_id in range(max_iter):
            sess.run(train_iter.initializer)
            batch_imgs, batch_labels = sess.run(train_next_element)
            # print(batch_imgs, batch_labels)
            acc, loss, _ = sess.run([acc_op, loss_op, train_op], feed_dict={X: batch_imgs, Y: batch_labels})

            if iter_id % DISPLAY_STEP == 0:

                train_acc_sum = 0.0
                train_loss_sum = 0.0
                sess.run(train_iter.initializer)
                for i in range(train_batch_num):
                    train_batch_imgs, train_batch_labels = sess.run(train_next_element)
                    train_acc, train_loss = sess.run([acc_op, loss_op], feed_dict={X: train_batch_imgs, Y: train_batch_labels})
                    # print(val_acc, val_loss, type(val_acc))
                    i_size = train_set_size % BATCH_SIZE if i == train_batch_num - 1 else BATCH_SIZE
                    train_acc_sum += train_acc[0] * i_size
                    train_loss_sum += train_loss * i_size
                train_acc_mean = train_acc_sum * 1.0 / train_set_size
                train_loss_mean = train_loss_sum * 1.0 / train_set_size

                val_acc_sum = 0.0
                val_loss_sum = 0.0
                sess.run(val_iter.initializer)  # outer can loop once data.
                for i in range(val_batch_num):
                    val_batch_imgs, val_batch_labels = sess.run(val_next_element)
                    val_acc, val_loss = sess.run([acc_op, loss_op], feed_dict={X: val_batch_imgs, Y: val_batch_labels})
                    i_size = val_set_size % BATCH_SIZE if i == val_batch_num - 1 else BATCH_SIZE
                    val_acc_sum += val_acc[0] * i_size
                    val_loss_sum += val_loss * i_size
                val_acc_mean = val_acc_sum * 1.0 / val_set_size
                val_loss_mean = val_loss_sum * 1.0 / val_set_size

                print("Step {}:".format(iter_id) +
                      "Minibatch Loss= {:.4f}".format(loss) +
                      ", train loss = {:.4f}".format(train_acc_mean) +
                      ", Training Accuracy= {:.3f}".format(train_acc_mean) +
                      ", Val Loss= {:.4f}".format(val_loss_mean) +
                      ", Val Accuracy= {:.3f}".format(val_acc_mean))

        saver = tf.train.Saver().save(sess, './tmp_logs/MyModel', global_step=max_iter)
    print("Optimization Finished!")

