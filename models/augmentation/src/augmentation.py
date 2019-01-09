#coding=utf-8

# -*- coding:utf-8 -*-
#!/usr/bin/env python

'''
#########################################################
enhance image data, create more dogs pciture to classify.
#########################################################
'''

import os
import random
import string
import datetime
import tensorflow as tf
from itertools import islice
import cv2


def RandEnhancePicture(raw_image, selected=1):

    #image_decode_jpeg = tf.image.decode_jpeg(image)
    #image_decode_jpeg = tf.image.convert_image_dtype(image_decode_jpeg, dtype=tf.float32) # convert image dtype to float
    print(raw_image.shape)
    image_decode_jpeg = tf.placeholder("uint8", [None, None, 3])
    rand = selected
    # rand = random.randint(1,7) # we only use 7 image Ops.
    tag = None
    if rand == 1: # flip up down
        image_flip_up_down = tf.image.flip_up_down(image_decode_jpeg)
        image_flip_up_down = tf.image.convert_image_dtype(image_flip_up_down, dtype=tf.uint8)
        img = image_flip_up_down
        tag = "vertical_flip"
    if rand == 2: # flip left right
        image_flip_left_right = tf.image.flip_left_right(image_decode_jpeg)
        image_flip_left_right = tf.image.convert_image_dtype(image_flip_left_right, dtype=tf.uint8)
        img = image_flip_left_right
        tag = "horizontal_flip"
    if rand == 3: # random adjust brightness
        image_random_brightness = tf.image.random_brightness(image_decode_jpeg, max_delta=0.01)
        image_random_brightness = tf.image.convert_image_dtype(image_random_brightness, dtype=tf.uint8)
        img = image_random_brightness
        tag = "adjust_brightness"
    if rand == 4: # random adjust contrast
        image_random_contrast = tf.image.random_contrast(image_decode_jpeg, 0.8, 1)
        image_random_contrast = tf.image.convert_image_dtype(image_random_contrast, dtype=tf.uint8)
        img = image_random_contrast
        tag = "adjust_contrast"
    if rand == 5: # random adjust hue
        image_random_hue = tf.image.random_hue(image_decode_jpeg, max_delta=0.05)
        image_random_hue = tf.image.convert_image_dtype(image_random_hue, dtype=tf.uint8)
        img = image_random_hue
        tag = "adjust_hue"
    if rand == 6: # random adjust saturation
        image_random_saturation = tf.image.random_saturation(image_decode_jpeg, 0.7, 1)
        image_random_saturation = tf.image.convert_image_dtype(image_random_saturation, dtype=tf.uint8)
        img = image_random_saturation

        tag = "adjust_saturation"
    if rand == 7: # adjust gamma
        image_adjust_gamma = tf.image.adjust_gamma(image_decode_jpeg, gamma=2)
        image_adjust_gamma = tf.image.convert_image_dtype(image_adjust_gamma, dtype=tf.uint8)
        # img = tf.image.encode_jpeg(image_adjust_gamma)
        img = image_adjust_gamma
        tag = "adjust_gamma"

    with tf.Session() as sess:  # create tensorflow session
        img = sess.run(img, feed_dict={image_decode_jpeg, raw_image})
    # tf.get_default_graph().finalize()
    return img, tag


def main():
    path = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/before_augmented"
    to_path = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/after_augmented"
    for e in os.listdir(path):
        f = os.path.join(path, e)
        image = cv2.imread(f)
        for selected in range(8):
            if selected == 0:
                img, tag = image, "raw"
            else:
                img, tag = RandEnhancePicture(image, selected=selected)
            output_f = os.path.join(to_path, "{}-{}.jpg".format(e.replace('.jpg', ''), tag))
            cv2.imwrite(output_f, img)



if __name__ == "__main__":
    print("begin to enhance picture data!!!")
    main()
    print("end of enhance picture data, good luck!!!")
