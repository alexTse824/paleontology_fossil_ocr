import os
import time

# directory settings
DIR_base = os.path.split(__file__)[0]
DIR_data = os.path.join(DIR_base, 'data')
DIR_straitified_dataset = os.path.join(DIR_data, "straitified_dataset")

# MongoDB settings
DB_address_ip = 'localhost'
DB_address_port = 27017
DB_name = 'paleontology_fossils_dataset'

# timestamp
CURRENT_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
