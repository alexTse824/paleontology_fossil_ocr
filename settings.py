import os
import time

# directory settings
DIR_base = os.path.split(os.path.abspath(__file__))[0]
DIR_data = os.path.join(DIR_base, 'data')
DIR_straitified_dataset = os.path.join(DIR_data, 'straitified_dataset')
DIR_weight = os.path.join(DIR_data, 'weight')
DIR_model_plot = os.path.join(DIR_data, 'model_plot')
DIR_binariazation = os.path.join(DIR_data, 'binarization')
DIR_log = os.path.join(DIR_data, 'log')

# timestamp
CURRENT_TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())
