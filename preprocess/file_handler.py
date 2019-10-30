import os
import re
import json
import cv2


def file_name_format(path):
    '''
    change filename in target directory to serialized number;
    modify file extension to *.jpg;
    '''
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


def get_ds_structure(ds_dir):
    '''get dataset structure and sort in dict'''
    ds_struct = {}
    for root, _, filenames in os.walk(ds_dir):
        for filename in filenames:
            if filename.split('.')[-1] == 'jpg':
                label = os.path.split(root)[-1]
                data_path = os.path.join(root, filename)
                try:
                    ds_struct[label].append(data_path)
                except KeyError:
                    ds_struct[label] = [data_path]
    return ds_struct


def output_dataset_info(ds_dir):
    '''organize *.jpg file in dataset information and output into json file'''
    ds_info = get_ds_structure(ds_dir)
    ds_info_json_file = os.path.join(os.path.dirname(ds_dir),
                                     f'{os.path.basename(ds_dir)}.json')

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
    
    # scale img to least w/h ratio
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    if new_w % 2 != min_side % 2:
        new_w -= 1
    if new_h % 2 != min_side % 2:
        new_h -= 1
    resize_img = cv2.resize(img, (new_w, new_h))

    # padding image to fill target width / height
    top = int((min_side - new_h) / 2),
    bottom = int((min_side - new_h) / 2)
    left = int((min_side - new_w) / 2)
    right = int((min_side - new_w) / 2)

    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imwrite(filename, pad_img)
