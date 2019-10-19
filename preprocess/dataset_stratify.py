from sklearn.model_selection import StratifiedKFold

from preprocess.db_handler import get_dataset_info


def dataset_stratified():
    X, y = get_dataset_info()
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    straitify_ret = {dataset_label: {} for dataset_label in y}

    for index, dataset_label in enumerate(y):
        for train_set, test_set in skf.split(X[index], [dataset_label]*len(X[index])):
            straitify_ret[dataset_label]['train'] = [X[index][i] for i in train_set]
            straitify_ret[dataset_label]['test'] = [X[index][i] for i in test_set]

    return straitify_ret
