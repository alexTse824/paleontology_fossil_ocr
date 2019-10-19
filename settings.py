import os
import time

# directory settings
DIR_base = os.path.split(__file__)[0]
DIR_data = os.path.join(DIR_base, 'data')
DIR_straitifiy = os.path.join(DIR_data, "straitifiy")

# MongoDB settings
DB_address_ip = 'localhost'
DB_address_port = 27017
DB_name = 'paleontology_fossils_dataset'

# timestamp
CURRENT_TIME = time.strftime("%Y%m%d_%H%M%S", time.localtime())
