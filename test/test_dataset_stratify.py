import unittest

from preprocess.dataset_stratify import dataset_stratified, stratify_ds_tfrecords


class TestDatasetStratify(unittest.TestCase):
    def test_dataset_stratified(self):
        ds_json = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/raw_data.json'
        dataset_stratified(ds_json, 5)

    def test_stratify_ds_tfrecords(self):
        ds_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/raw_data_224'
        output_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/tfrecords_data/raw_data_224_tfs'
        stratify_ds_tfrecords(ds_dir, output_dir)


if __name__ == '__main__':
    unittest.main()
