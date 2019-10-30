import unittest
import json

from utils.export_svg_chart import export_dataset_stratified_hist, export_train_val_svg


class TestUtils(unittest.TestCase):
    def test_export_dataset_stratified_hist(self):
        train_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/data/0/train'
        validation_dir = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/data/0/validation'
        title = 'Stratified dataset in KFold.5'
        output_path = '/Users/xie/Desktop/KFold.5.stratified.svg'
        export_dataset_stratified_hist(train_dir, validation_dir, title, output_path)

    def test_export_train_val_svg(self):
        # data structure: [[time1, epoch1, acc1], [time2, epoch2, acc2], ...]
        train_loss_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/graph_data/KFold.5.SET0.BATCH64.EPOCHS100.loss.json'
        val_loss_data_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/graph/graph_data/KFold.5.SET0.BATCH64.EPOCHS100.val_loss.json'

        with open(train_loss_data_file) as f:
            train_loss_data = json.load(f)
        with open(val_loss_data_file) as f:
            val_loss_data = json.load(f)

        export_train_val_svg('Train/Validation loss of KFold.5.SET0.B64.E100',
                             'Loss',
                             [i[1] for i in train_loss_data],
                             [i[2] for i in train_loss_data],
                             [i[2] for i in val_loss_data],
                             '/Users/xie/Desktop/KFold.5.SET0.B64.E100.train_val_loss.svg')


if __name__ == '__main__':
    unittest.main()
