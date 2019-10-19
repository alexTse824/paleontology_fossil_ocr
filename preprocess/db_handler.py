import os
from pymongo import MongoClient

from settings import DB_address_ip, DB_address_port, DB_name


def create_dataset_info(ds_dir, db_name=DB_name):
    '''
    insert dataset pictures into mongodb;
    path: dataset's directory absolute path, with format: */*/label1/0.jpg;
    '''
    print(f'Processing {ds_dir}...')
    db = get_mongo_connection(db_name)

    label = os.path.split(ds_dir)[-1]
    collection = db[label]

    data = [{
        'name': os.path.splitext(file_name)[0],
        'file_path': os.path.join(ds_dir, file_name)
    } for file_name in os.listdir(ds_dir)]

    return collection.insert_many(data)


def get_dataset_info(db_name=DB_name):
    '''
    get target dataset overall infomation from db
    return: (
        [[class0_1.jpg, class0_2.jpg], [class1_1.jpg, class1_2.jpg], ...],
        [class0, class1, ...]
    )
    '''
    db = get_mongo_connection(db_name)

    collections = [collection for collection in db.list_collection_names()]
    datasets = []
    for collection in collections:
        datasets.append(list(document['file_path']
                        for document in db[collection].find({})))

    return [datasets, collections]


def get_mongo_connection(db_name=DB_name):
    client = MongoClient(DB_address_ip, DB_address_port)
    print(f'Connect to {DB_name}')
    return client[db_name]
