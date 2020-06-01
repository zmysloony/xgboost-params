import os.path
from zipfile import ZipFile
import pandas as pd
import algorithms


def unpack_data():
    if not os.path.exists('data/porto-seguro-safe-driver-prediction.zip'):
        print('Download data from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data and put it in '
              '/data folder.')
        return False
    with ZipFile('data/porto-seguro-safe-driver-prediction.zip') as zipped:
        print('Extracting...')
        zipped.extractall('data/')
        print('Extracted successfully.')
    return True


def check_data_files():
    if not os.path.exists('data/train.csv'):
        return unpack_data()
    return True


# data is 100% complete, no missing attributes
# max rows = 595212
def get_train(rows=10000):
    train_data = pd.read_csv('data/train.csv', nrows=rows, header=0)
    train_target = train_data['target']
    train_data = train_data.drop(['id', 'target'], axis=1)
    return train_data, train_target


def get_train_df():
    return pd.read_csv('data/train.csv', header=0)


if __name__ == '__main__':
    if not check_data_files():
        exit(1)
    # data_analysis.analyse_dataset(get_train_df())
    # data, target = get_train(50000)

    driver_data_clean = get_train_df().head(80000)

    # driver_data_clean = data_analysis.replace_to_nan(driver_data_clean)

    driver_data_clean = driver_data_clean.drop('ps_car_03_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_car_05_cat', axis=1)

    target = driver_data_clean['target']
    data = driver_data_clean.drop(['id', 'target'], axis=1)

    # algorithms.perform_brute_force(data, target)
    best_param_set = algorithms.perform_hill_climbing(data, target)
    print(best_param_set)
    # param = {
    #     # TODO : params for search - will be prepared in search algorithm functions
    #     'max_depth': 10,  # the maximum depth of each tree
    #     'eta': 0.05,  # the training step for each iteration
    #     'silent': 1,  # logging mode - quiet
    #
    #     'objective': 'binary:hinge',
    #     # 'num_class': 2,  # the number of classes that exist in this datset
    #     # NOTE : num_class not used in binary classification
    #     'eval_metric': 'auc',  # use area under precision & recall curve as eval_metric
    #     # NOTE : we might want to change evaluation to ROC curve as 'auc' is directly Area under ROC curve metric
    #     # NOTE : might be a parameter to optimize
    #     'tree_method': 'hist',
    #     'nthread': 0
    # }
    # mdl1 = model.train(data, target, param, 10, 4)


