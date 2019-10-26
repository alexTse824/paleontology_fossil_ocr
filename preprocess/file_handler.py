import sys
sys.path.append('.')

import os
import re
import json

from settings import DIR_data


def file_name_format(path):
    """change filename in target directory to serialized number and with extension .jpg"""
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


def output_dataset_info(ds_path):
    '''organize *.jpg file in dataset information and output into json file'''
    ds_info = {}
    for root, _, filenames in os.walk(ds_path):
        for filename in filenames:
            if filename.split('.')[-1] == 'jpg':
                label = os.path.split(root)[-1]
                data_path = os.path.join(root, filename)
                try:
                    ds_info[label].append(data_path)
                except KeyError:
                    ds_info[label] = [data_path]
    ds_info_json_file = os.path.join(ds_path,
                                     f'{os.path.split(ds_path)[-1]}.json')
    with open(ds_info_json_file, 'w') as f:
        json.dump(ds_info, f, indent=4)


def get_dataset_info(json_filepath):
    '''get dataset info from json file'''
    with open(json_filepath) as f:
        return json.load(f)




if __name__ == "__main__":
    # format picture names under /data/raw_data
    # for i in range(9):
    # file_name_format(os.path.join(DIR_data, 'raw_data', f'class_{i}'))

    # output_dataset_info(os.path.join(DIR_data, 'raw_data'))

    get_dataset_info(
        '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/raw_data.json'
    )
