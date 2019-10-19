from sklearn.model_selection import StratifiedKFold

from preprocess.db_handler import get_dataset_info
from preprocess.file_handler import dataset_file_straitify
from settings import CURRENT_TIME


def dataset_stratified(kfold_split_num=5):
    X, y = get_dataset_info()
    skf = StratifiedKFold(n_splits=kfold_split_num, shuffle=True)
    straitify_ret = {dataset_label: {} for dataset_label in y}

    for index, dataset_label in enumerate(y):
        for validation_set, test_set in skf.split(X[index], [dataset_label]*len(X[index])):
            straitify_ret[dataset_label]['validation'] = [X[index][i] for i in validation_set]
            straitify_ret[dataset_label]['test'] = [X[index][i] for i in test_set]

    validation_dir_name = f'KFold.{kfold_split_num}.{CURRENT_TIME}'
    dataset_file_straitify(validation_dir_name, straitify_ret)

    return straitify_ret
