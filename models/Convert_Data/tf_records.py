# make_FTRecords.py

import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到

# 制作二进制FTRecord文件
# refer to https://zhuanlan.zhihu.com/p/35666271
# 获取当前地址
cwd = "/Users/jiangyy/DataSet/rib_dataSet/first20labeled/data_augmentation/testSet"
classes = {'fragment','unfragment'} # 设置为两类
writer = tf.python_io.TFRecordWriter("/Users/jiangyy/projects/temp/medical-rib/models/temp_data/test_bone.tfrecords") # 设置我们要生成的文件

for index, name in enumerate(classes):
    # 构建路径名的时候，注意"/" "\"的使用
    # 在Linux中以"/"分割，在Windows下以"\"分割，各位根据自己情况修改
    class_path = cwd +"/"+ name+"/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name # 构建出每组数据的地址

        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes() #将图片转化为二进制的格式

        # example对象对label和image数据进行封装
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=
                                                                              [img_raw]))
                }
            )
        )
        # 将序列转为字符串
        writer.write(example.SerializeToString())
writer.close()