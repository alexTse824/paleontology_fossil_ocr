import os
import re


def file_name_format(path):
    '''change filename in target directory to serialized number'''
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


def generate_ds_label_path(path):
    '''
    generate dataset's label and abspath info
    yield: (label, img_abspath)
    '''
    label = os.path.split(path)[-1]
    for file_name in os.listdir(path):
        yield(label, os.path.join(path, file_name))
