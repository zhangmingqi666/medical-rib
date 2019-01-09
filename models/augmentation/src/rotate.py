import tensorflow as tf
import numpy as np
import os
import cv2

def rotate_images(img, selected):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, None, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for i in range(3):  # Rotation at 90, 180 and 270 degrees
        rotated_img = sess.run(tf_img, feed_dict={X: img, k: selected})
        X_rotate.append(rotated_img)
    return X_rotate[0]


def flip_images(img, selected):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, None, 3))
    k = tf.placeholder(tf.int32)
    if selected == 1:
        tf_img = tf.image.flip_left_right(X)
        tag = "flip_left_right"
    elif selected == 2:
        tf_img = tf.image.flip_up_down(X)
        tag = "flip_up_down"
    else:
        tf_img = tf.image.transpose_image(X)
        tag = "transpose"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # for i in range(3):  # Rotation at 90, 180 and 270 degrees
        rotated_img = sess.run(tf_img, feed_dict={X: img, k: selected})
        X_rotate.append(rotated_img)
    return X_rotate[0], tag


path = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/before_augmented"
to_path = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/after_augmented"
for e in os.listdir(path):
    f = os.path.join(path, e)
    image = cv2.imread(f)
    for selected in range(1, 4, 1):
        #if selected == 0:

        img, tag = flip_images(image, selected)
        output_f = os.path.join(to_path, "{}-{}.jpg".format(e.replace('.jpg', ''), tag))
        cv2.imwrite(output_f, img)
#rotated_imgs = rotate_images(X_imgs)