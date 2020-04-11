import os.path
from zipfile import ZipFile
import pandas as pd
import model
import data_analysis


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
    d = pd.read_csv('data/train.csv', nrows=rows, header=0)
    targt = d['target']
    d = d.drop(['id', 'target'], axis=1)
    return d, targt


def get_train_df():
    return pd.read_csv('data/train.csv', header=0)


if __name__ == '__main__':
    if not check_data_files():
        exit(1)
    data_analysis.analyse_dataset(get_train_df())
    data, target = get_train(10000)
    mdl = model.train(data, target)
