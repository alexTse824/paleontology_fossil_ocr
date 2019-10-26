import sys
sys.path.append('.')
import os
from sklearn.model_selection import StratifiedKFold

from db_handler import get_dataset_info
from file_handler import dataset_file_straitify
from settings import CURRENT_TIME


def dataset_stratified(kfold_split_num=5):
    X, y = get_dataset_info()
    skf = StratifiedKFold(n_splits=kfold_split_num, shuffle=True)

    stratified_dir_name = f'KFold.{kfold_split_num}.{CURRENT_TIME}'

    for label_index, dataset_label in enumerate(y):
        # skf.split para1: ['class01/1.jpg', 'class01/2.jpg', ...]
        # skf.split para2: ['class01', 'class01', ...]
        for set_index, (train_set, validation_set) in enumerate(
                skf.split(X[label_index],
                          [dataset_label] * len(X[label_index]))):

            current_stratified_dir_name = os.path.join(stratified_dir_name,
                                                       str(set_index))

            dataset_file_straitify(
                current_stratified_dir_name, {
                    dataset_label: {
                        'train': [X[label_index][i] for i in train_set],
                        'validation':
                        [X[label_index][i] for i in validation_set],
                    }
                })
