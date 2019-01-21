import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg

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


train_log_dir = "tmp_logs"
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

train_data_path="/Users/jiangyy/projects/temp/medical-rib/models/temp_data/train_bone.tfrecords"
test_data_path="/Users/jiangyy/projects/temp/medical-rib/models/temp_data/test_bone.tfrecords"

with tf.Graph().as_default():
    # Set up the data loading:
    train_images, train_labels = load_tfrecord(tf_records_file_name=train_data_path, channel=3)

    images, labels, = tf.train.batch([train_images, train_labels], batch_size=32)
    # print(images)
    # exit(0)
    test_images, test_labels = load_tfrecord(tf_records_file_name=test_data_path, channel=3)

    # Define the model:
    predictions = vgg.vgg_16(images, num_classes=2, is_training=True)

    # print(predictions)

    # Specify the loss function:
    slim.losses.softmax_cross_entropy(predictions, labels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    # train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    # slim.learning.train(train_tensor, train_log_dir)