import sys
import os
import shutil
from sklearn.model_selection import StratifiedKFold

sys.path.append('.')
from file_handler import get_dataset_info
from settings import DIR_straitified_dataset


def sort_stratified_dataset(dir_name, label, train_set, validation_set):
    '''copy dataset files to matching directories according to given stratified_data'''
    current_straitified_dir = os.path.join(DIR_straitified_dataset, dir_name)
    train_set_dir = os.path.join(current_straitified_dir, 'train', label)
    validation_set_dir = os.path.join(current_straitified_dir, 'validation', label)

    os.makedirs(train_set_dir)
    os.makedirs(validation_set_dir)

    for train_ds_file in train_set:
        file_name = os.path.split(train_ds_file)[-1]
        shutil.copyfile(train_ds_file, os.path.join(train_set_dir, file_name))

    for validation_ds_file in validation_set:
        file_name = os.path.split(validation_ds_file)[-1]
        shutil.copyfile(validation_ds_file, os.path.join(validation_set_dir, file_name))


def dataset_stratified(ds_info_file, kfold_split_num=5):
    ds_info = get_dataset_info(ds_info_file)
    skf = StratifiedKFold(n_splits=kfold_split_num, shuffle=True)

    stratified_dir_name = f'KFold.{kfold_split_num}'

    for label, dataset_list in ds_info.items():
        # skf.split para1: ['class01/1.jpg', 'class01/2.jpg', ...]
        # skf.split para2: ['class01', 'class01', ...]
        for set_index, (train_set, validation_set) in enumerate(skf.split(dataset_list, [label] * len(dataset_list))):
            current_stratified_dir_name = os.path.join(stratified_dir_name, str(set_index))
            sort_stratified_dataset(current_stratified_dir_name, label, [dataset_list[i] for i in train_set], [dataset_list[i] for i in validation_set])


if __name__ == "__main__":
    dataset_stratified('/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/raw_data.json', 5)
