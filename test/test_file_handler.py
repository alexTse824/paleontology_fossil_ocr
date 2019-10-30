import unittest
import os

from preprocess.file_handler import file_name_format, output_dataset_info, img_resize
from settings import DIR_data


class TestFileHandler(unittest.TestCase):
    def test_file_name_format(self):
        # format picture names under /data/raw_data
        for i in range(9):
            file_name_format(os.path.join(DIR_data, 'raw_data', f'class_{i}'))

    def test_output_dataset_info(self):
        output_dataset_info(os.path.join(DIR_data, 'raw_data_448'))

    def test_img_resiza(self):
        path = '/Users/xie/Code/paleontology_fossil_ocr/data/raw_data_448'
        for root, dirnames, filenames in os.walk(path):
            for file in filenames:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[-1] == '.jpg':
                    img_resize(file_path, 448)


if __name__ == '__main__':
    unittest.main()
