import os
import unittest

from utils.file_handler import file_name_format, generate_ds_label_path


class TestUtils(unittest.TestCase):
    def test_file_name_format(self):
        target_dir = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/class_0'

        for i in range(9):
            current_path = os.path.join(target_dir, f'class_{i}')
            file_name_format(current_path)

    def test_generate_ds_label_path(self):
        path = '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/class_0'
        for i in generate_ds_label_path(path):
            print(i)


if __name__ == "__main__":
    unittest.main()
