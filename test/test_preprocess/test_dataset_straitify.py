import unittest

from preprocess.dataset_stratify import dataset_stratified


class TestDataStraitify(unittest.TestCase):
    def test_dataset_stratified(self):
        ret = dataset_stratified()
        for key, value in ret.items():
            print(f'Label: {key}')
            print(f'train_set number: {len(value["train"])}')
            print(f'test_set number: {len(value["test"])}')
            print('-' * 50)


if __name__ == "__main__":
    unittest.main()
