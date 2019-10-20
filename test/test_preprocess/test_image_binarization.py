import unittest

from preprocess.image_binarization import global_threshold_binarization


class TestImageBinarization(unittest.TestCase):
    def test_global_threshold_binarization(self):
        image = '01_8_CH195_large.jpg'
        global_threshold_binarization(image)


if __name__ == "__main__":
    unittest.main()
