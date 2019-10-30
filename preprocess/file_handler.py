import sys
sys.path.append('.')

import os
import re
import json
import cv2
import glob

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
    ds_info_json_file = os.path.join(os.path.dirname(ds_path),
                                     f'{os.path.basename(ds_path)}.json')
    with open(ds_info_json_file, 'w') as f:
        json.dump(ds_info, f, indent=4)


def get_dataset_info(json_filepath):
    '''get dataset info from json file'''
    with open(json_filepath) as f:
        return json.load(f)


def img_resize(filename, min_side=224):
    '''scale image and padding white to fit target width & height'''
    img = cv2.imread(filename)
    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    if new_w % 2 != min_side % 2:
        new_w -= 1
    if new_h % 2 != min_side % 2:
        new_h -= 1
    resize_img = cv2.resize(img, (new_w, new_h))

    top, bottom, left, right = int((min_side - new_h) / 2), int((min_side - new_h) / 2), int((min_side - new_w) / 2), int((min_side - new_w) / 2)

    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imwrite(filename, pad_img)


if __name__ == "__main__":
    # format picture names under /data/raw_data
    # for i in range(9):
        # file_name_format(os.path.join(DIR_data, 'raw_data', f'class_{i}'))  

    output_dataset_info(os.path.join(DIR_data, 'raw_data_448'))

    # get_dataset_info(
    #     '/Users/xie/Code/NJU/paleontology_fossil_ocr/data/raw_data/raw_data.json'
    # )

    # path = '/Users/xie/Code/paleontology_fossil_ocr/data/raw_data_448'
    # for root, dirnames, filenames in os.walk(path):
    #     for file in filenames:
    #         file_path = os.path.join(root, file)
    #         if os.path.splitext(file_path)[-1] == '.jpg':
    #             img_resize(file_path, 448)

