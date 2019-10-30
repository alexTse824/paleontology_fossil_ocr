import unittest

from preprocess.dataset_stratify import dataset_stratified


class TestDatasetStratify(unittest.TestCase):
    def test_dataset_stratified(self):
        dataset_stratified('/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/raw_data.json', 5)


if __name__ == '__main__':
    unittest.main()
