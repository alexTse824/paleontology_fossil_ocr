import unittest
import os
from pprint import pprint

from preprocess.db_handler import create_dataset_info, get_dataset_info


class TestDB(unittest.TestCase):
    def test_create_dataset_info(self):
        target_dir = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data'

        for i in range(9):
            ds_dir = os.path.join(target_dir, f'class_{i}')
            create_dataset_info(ds_dir)

    def test_get_dataset_info(self):
        ret = get_dataset_info()
        pprint(ret)


if __name__ == "__main__":
    unittest.main()
