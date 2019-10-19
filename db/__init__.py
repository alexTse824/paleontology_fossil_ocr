import os
from pymongo import MongoClient

from settings import DB_address_ip, DB_address_port, DB_name


def create_dataset_info(ds_dir):
    '''
    insert dataset pictures into mongodb;
    path: dataset's directory absolute path, with format: */*/label1/0.jpg;
    '''
    print(f'Processing {ds_dir}...')
    client = MongoClient(DB_address_ip, DB_address_port)
    db = client[DB_name]
    label = os.path.split(ds_dir)[-1]
    collection = db[label]

    data = [{
        'name': os.path.splitext(file_name)[0],
        'file_path': os.path.join(ds_dir, file_name)
    } for file_name in os.listdir(ds_dir)]

    return collection.insert_many(data)
