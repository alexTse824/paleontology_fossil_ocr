'''Convert dataset file into TFRecords format'''
import json
import tensorflow as tf
from PIL import Image


def make_tfrecord_dataset(json_path, tfrecords_path):
    json_path = json_path
    with open(json_path) as f:
        ds_info = json.load(f)

    writer = tf.io.TFRecordWriter(tfrecords_path)

    for label, files in ds_info.items():
        for file in files:
            img = Image.open(file)
            img_raw = img.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label)])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))

            writer.write(example.SerializeToString())

    writer.close()


def conver_2_tfrecords(label, data_set, tfrecord_file_path):
    '''
    label: class_1
    dataset: [class_1/1.jpg, class_1/2.jpg, ...]
    '''
    writer = tf.io.TFRecordWriter(tfrecord_file_path)

    for image in data_set:
        img = Image.open(image)
        img_raw = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label)])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

    writer.close()
