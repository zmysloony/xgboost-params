import os.path
from zipfile import ZipFile
import pandas as pd
import model


def unpack_data():
    with ZipFile('data/porto-seguro-safe-driver-prediction.zip') as zipped:
        print('Extracting...')
        zipped.extractall('data/')
        print('Extracted successfully.')


def check_data_files():
    if not (os.path.exists('data/sample_submission.csv') and os.path.exists('data/test.csv') and os.path.exists('data/train.csv')):
        unpack_data()


# data is 100% complete, no missing attributes
# max rows = 595212
def get_train(rows=10000):
    d = pd.read_csv('data/train.csv', nrows=rows, header=0)
    targt = d['target']
    d = d.drop(['id', 'target'], axis=1)
    return d, targt


if __name__ == '__main__':
    check_data_files()
    data, target = get_train(10000)
    mdl = model.train(data, target)
