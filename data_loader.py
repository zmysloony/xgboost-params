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


# max rows = 595212
def get_train(rows=10000):
    train_data = pd.read_csv('data/train.csv', nrows=rows, header=0)
    train_target = train_data['target']
    train_data = train_data.drop(['id', 'target'], axis=1)
    return train_data, train_target


def get_train_df():
    return pd.read_csv('data/train.csv', header=0)