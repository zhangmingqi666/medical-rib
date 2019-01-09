import os
import tensorflow as tf
from PIL import Image

# img path
BONE_IMG_PATH = '/home/jiangyy/Desktop/temp_data/JPEGSImages'
ANNATATION_PATH = '/home/jiangyy/Desktop/temp_data/Annatations'

def read_img_name_and_label(imgs_path, label_path):
    """
    """
    # read jpg name
    img_name = os.listdir(imgs_path)
    # cut '.jpg'
    for i in range(len(img_name)):
        img_name[i] = img_name[i][:-4]

    # read fracture bone's xml file name
    label_list = os.listdir(label_path)
    # cut '.xml'
    for i in range(len(label_list)):
        label_list[i] = label_list[i][:-4]

    # set data dictionary containing img name and label
    data_dict = {}
    for img in img_name:
        if img in label_list:
            data_dict[img] = 1
        else:
            data_dict[img] = 0

    return list(data_dict.keys()), list(data_dict.values())

def save_data_to_tf_records(imgs_path, label_path, tf_records_file_name='train.tfrecords'):
    """
    convert source 17flowers data set to TFrecords
    :param data_path: 17flowers images file path
    :param tf_records_file_name: TFrecords file name
    :return: None
    """
    writer = tf.python_io.TFRecordWriter(tf_records_file_name)

    jpg_name, label = read_img_name_and_label(imgs_path, label_path)
    # read jpg
    for jpg, i in zip(jpg_name, label):
        img_path = os.path.join(imgs_path, jpg + '.jpg')
        img = Image.open(img_path)
        if i == 1:
            img = img.resize((224, 224))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # write TFrecords file
            writer.write(example.SerializeToString())
        else:
            img = img.resize((224, 224))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # write TFrecords file
            writer.write(example.SerializeToString())
    writer.close()


save_data_to_tf_records(imgs_path=BONE_IMG_PATH, label_path=ANNATATION_PATH, tf_records_file_name='train.bone_tfrecords')


def read_and_decode(imgs_path, label_path, tf_records_file_name='train.bone_tfrecords'):
    """
    generate TFrecords file and read images and labels from TFrecords file
    :param data_path: 17flowers images file path
    :param tf_records_file_name: TFrecords file name
    :return img: 17flower images ([shape:[224, 224, 3]])
    :return label: 17flowers labels
    """
    save_data_to_tf_records(imgs_path, label_path, tf_records_file_name)
    filename_queue = tf.train.string_input_producer([tf_records_file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label


def load_tfrecord(tf_records_file_name='train.bone_tfrecords', channel=1):
    """
    read images and labels from TFrecords file
    :param tf_records_file_name: TFrecords file name
    :return img: 17flower images ([shape:[224, 224, 3]])
    :return label: 17flowers labels
    """
    filename_queue = tf.train.string_input_producer([tf_records_file_name])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, channel])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label
