import os
import shutil
from sklearn.model_selection import StratifiedKFold

from preprocess.file_handler import get_dataset_info, get_ds_structure
from preprocess.tfrecords_convert import conver_2_tfrecords
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


def stratify_dataset(ds_struct, n_splits=5):
    '''
    ds_struct: {'class1': [class1/1.jpg, ...]...}
    return: {
        0: {
            'train': {
                'class1': [class1/1.jpg, ...],
                ...
            },
            'validation': {
                'class1': [class1/10.jpg, ...],
                ...
            }
        },
        1: {...},
        ...
    }
    '''
    stratified_ds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for label, dataset in ds_struct.items():
        label_list = [label] * len(dataset)
        # skf.split para1: ['class01/1.jpg', 'class01/2.jpg', ...]
        # skf.split para2: ['class01', 'class01', ...]
        kfold_gen = skf.split(dataset, label_list)
        for set_index, (train_set, validation_set) in enumerate(kfold_gen):
            set_index = str(set_index)
            if set_index not in stratified_ds.keys():
                stratified_ds[set_index] = {'train': {}, 'validation': {}}
            stratified_ds[set_index]['train'][label] = [dataset[i] for i in train_set]
            stratified_ds[set_index]['validation'][label] = [dataset[i] for i in validation_set]

    return stratified_ds


def dataset_stratified(ds_info_file, kfold_split_num=5):
    ds_info = get_dataset_info(ds_info_file)
    skf = StratifiedKFold(n_splits=kfold_split_num, shuffle=True)

    stratified_dir_name = f'KFold.{kfold_split_num}'

    for label, dataset_list in ds_info.items():
        # skf.split para1: ['class01/1.jpg', 'class01/2.jpg', ...]
        # skf.split para2: ['class01', 'class01', ...]
        for set_index, (train_set, validation_set) in enumerate(skf.split(dataset_list, [label] * len(dataset_list))):
            current_stratified_dir_name = os.path.join(stratified_dir_name, str(set_index))
            sort_stratified_dataset(current_stratified_dir_name, label, [dataset_list[i] for i in train_set],
                                    [dataset_list[i] for i in validation_set])


def stratify_ds_tfrecords(ds_dir, tfrecords_dir, n_splits=5):
    '''same as dataset_stratified, but convert into tfrecords format'''
    ds_struct = get_ds_structure(ds_dir)
    stratified_ds = stratify_dataset(ds_struct)

    for set_index, set_dataset in stratified_ds.items():
        set_train_dir = os.path.join(tfrecords_dir, set_index, 'train')
        set_validation_dir = os.path.join(tfrecords_dir, set_index, 'validation')
        if not os.path.exists(set_train_dir):
            os.makedirs(set_train_dir)
        if not os.path.exists(set_validation_dir):
            os.makedirs(set_validation_dir)

        for set_type, set_dir in [('train', set_train_dir), ('validation', set_validation_dir)]:
            for label, current_set in set_dataset[set_type].items():
                tfrecord_file_path = os.path.join(set_dir, f'{label}.tfrecords')
                conver_2_tfrecords(label, current_set, tfrecord_file_path)
