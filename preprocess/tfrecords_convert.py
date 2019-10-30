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


if __name__ == '__main__':
    make_tfrecord_dataset('/Users/xie/Code/paleontology_fossil_ocr/data/raw_data_448/raw_data_448.json',
                          '/Users/xie/Code/paleontology_fossil_ocr/data/Fossil9.448.tfrecords')
