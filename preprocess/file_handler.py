import os
import shutil

from settings import DIR_straitifiy


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
    current_train_dir = os.path.join(DIR_straitifiy, dir_name)

    for label, data_set in stratified_data.items():
        label_dir = os.path.join(current_train_dir, label)
        train_dir = os.path.join(label_dir, 'train')
        test_dir = os.path.join(label_dir, 'test')
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        for train_file in data_set['train']:
            file_name = os.path.split(train_file)[-1]
            shutil.copyfile(train_file, os.path.join(train_dir, file_name))

        for test_file in data_set['test']:
            file_name = os.path.split(test_file)[-1]
            shutil.copyfile(test_file, os.path.join(test_dir, file_name))
