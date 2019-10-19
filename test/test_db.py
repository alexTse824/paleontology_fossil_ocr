import unittest
import os

from db import create_dataset_info


class TestDB(unittest.TestCase):
    def test_create_dataset_info(self):
        target_dir = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data'

        for i in range(9):
            ds_dir = os.path.join(target_dir, f'class_{i}')
            create_dataset_info(ds_dir)


if __name__ == "__main__":
    unittest.main()
