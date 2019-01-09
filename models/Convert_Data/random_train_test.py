
import os
input_1 = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/after_augmented"
sample_prob=[0.8, 0.2]
import numpy as np

import os
import random
import argparse
import numpy as np
import sys


def get_all_data(xml_path=None):
    all_xml = os.listdir(xml_path)
    res = [e.replace('.jpg', '') for e in all_xml]
    return res


#def generate_train_val_test(data_list=None, sample_prob=[0.8, 0.2], main_path=None):



xml_path="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/negtive_images"

data_list = get_all_data(xml_path=xml_path)

sample_prob=[0.8, 0.2]
index = np.random.choice(len(sample_prob), len(data_list), p=sample_prob)
data_arr = np.array(data_list)
train_arr = data_arr[index == 0]
val_arr = data_arr[index == 1]

train_file="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/trainSet/unfragment.txt"
test_file="/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/testSet/unfragment.txt"

with open(train_file, 'w') as f:
    f.write('\n'.join(train_arr.tolist()))

with open(test_file, 'w') as f:
    f.write('\n'.join(val_arr.tolist()))
