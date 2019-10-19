import os

# directory settings
DIR_base = os.path.split(__file__)[0]
DIR_data = os.path.join(DIR_base, 'data')

# db settings
DB_address_ip = 'localhost'
DB_address_port = 27017
DB_name = 'paleontology_fossils_dataset'
