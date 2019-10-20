import unittest

from preprocess.dataset_stratify import dataset_stratified


class TestDataStraitify(unittest.TestCase):
    def test_dataset_stratified(self):
        dataset_stratified()


if __name__ == "__main__":
    unittest.main()
