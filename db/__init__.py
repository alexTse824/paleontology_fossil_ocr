import os
import pymysql

from settings import DB_username, DB_password, DB_address_ip, DB_name


def create_dataset_info(ds_dir):
    '''
    insert dataset info into db;
    path: dataset's directory absolute path, with format: */*/label1/0.jpg;
    '''
    db = pymysql.connect(DB_address_ip, DB_username, DB_password, DB_name)
    cursor = db.cursor()

    table_name = os.path.split(ds_dir)[-1]
    create_table_sql = 'CREATE TABLE {} (\
            id INT AUTO_INCREMENT, \
            file_path VARCHAR(256) NOT NULL, \
            PRIMARY KEY ( id ));'
    cursor.execute(create_table_sql.format(table_name))

    insert_data_sql = 'INSERT INTO {} (file_path) VALUES ("{}");'
    for file_name in os.listdir(ds_dir):
        file_path = os.path.join(ds_dir, file_name)
        cursor.execute(insert_data_sql.format(table_name, file_path))

    db.commit()
    cursor.close()
    db.close()

