'''Convert dataset file into TFRecords format'''
import json
import tensorflow as tf
from PIL import Image


def _parse_feature(label, image):
    img = Image.open(image)
    img_raw = img.tobytes()

    return {
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label)])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }


def _parse_record(tfre_file):
    file = tf.parse_single_example(tfre_file, features={
        'label': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
    })

    label = tf.decode_raw(file['label'], tf.float32)
    img = tf.reshape(tf.decode_raw(file['image'], tf.float32), shape=(224, 224, 3))

    return label, img


def make_tfrecord_dataset(json_path, tfrecords_path):
    json_path = json_path
    with open(json_path) as f:
        ds_info = json.load(f)

    writer = tf.io.TFRecordWriter(tfrecords_path)

    for label, files in ds_info.items():
        for file in files:
            example = tf.train.Example(features=tf.train.Features(feature=_parse_feature(label, file)))
            writer.write(example.SerializeToString())
    writer.close()


def conver_2_tfrecords(label, data_set, tfrecord_file_path):
    '''
    label: class_1
    dataset: [class_1/1.jpg, class_1/2.jpg, ...]
    '''
    writer = tf.io.TFRecordWriter(tfrecord_file_path)

    for image in data_set:
        example = tf.train.Example(features=tf.train.Features(feature=_parse_feature(label, image)))
        writer.write(example.SerializeToString())

    writer.close()


def convert2tfrecords_mixed(dataset, tfrecord_file_path):
    '''
    dataset: [(class_1, class_1/1.jpg), ...]
    '''
    writer = tf.io.TFRecordWriter(tfrecord_file_path)
    for (label, image_path) in dataset:
        example = tf.train.Example(features=tf.train.Features(feature=_parse_feature(label, image_path)))
        writer.write(example.SerializeToString())

    writer.close()

