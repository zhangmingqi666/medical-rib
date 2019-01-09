import tensorflow as tf
from tensorflow.contrib import layers

regularizer = layers.l1_regularizer(0.1)
with tf.variable_scope('var', initializer=tf.random_normal_initializer(),
regularizer=regularizer):
    weight = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())
with tf.variable_scope('var2', initializer=tf.random_normal_initializer(),
regularizer=regularizer):
    weight2 = tf.get_variable('weight', shape=[8], initializer=tf.ones_initializer())

regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
print(regularization_loss)