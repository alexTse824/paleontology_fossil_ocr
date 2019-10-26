import sys
sys.path.append('.')
import os
import re
import shutil

from settings import DIR_straitified_dataset, DIR_data


def file_name_format(path):
    """change filename in target directory to serialized number"""
    os.chdir(path)
    file_list = os.listdir(path)

    index = 0
    for file_name in file_list:
        pattern = re.findall('.+.(jpg|png|gif|JPG|jpeg)', file_name)
        if pattern:
            os.rename(file_name, f'{index}.jpg')
            index += 1
        else:
            print(file_name)


def dataset_file_straitify(dir_name, stratified_data):
    '''copy dataset files to matching directories according to given stratified_data'''
    current_straitified_dir = os.path.join(DIR_straitified_dataset, dir_name)
    train_set_dir = os.path.join(current_straitified_dir, 'train')
    validation_set_dir = os.path.join(current_straitified_dir, 'validation')

    for label, data_set in stratified_data.items():
        label_validation_dir = os.path.join(train_set_dir, label)
        label_test_dir = os.path.join(validation_set_dir, label)
        os.makedirs(label_validation_dir)
        os.makedirs(label_test_dir)

        for train_file in data_set['train']:
            file_name = os.path.split(train_file)[-1]
            shutil.copyfile(train_file,
                            os.path.join(label_validation_dir, file_name))

        for validation_file in data_set['validation']:
            file_name = os.path.split(validation_file)[-1]
            shutil.copyfile(validation_file,
                            os.path.join(label_test_dir, file_name))


# if __name__ == "__main__":
    # format picture names under /data/raw_data
    # for i in range(9):
        # file_name_format(os.path.join(DIR_data, 'raw_data', f'class_{i}'))
