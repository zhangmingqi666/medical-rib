import tensorflow as tf

import numpy as np
import tensorflow as tf
B = 5
N = 10
input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    c3 = sess.run(op)
    print(c3.shape)