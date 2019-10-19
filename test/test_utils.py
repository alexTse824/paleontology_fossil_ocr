import os
import unittest

from utils import file_name_format


class TestUtils(unittest.TestCase):
    def test_file_name_format(self):
        target_dir = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/class_0'

        for i in range(9):
            current_path = os.path.join(target_dir, f'class_{i}')
            file_name_format(current_path)


if __name__ == "__main__":
    unittest.main()
