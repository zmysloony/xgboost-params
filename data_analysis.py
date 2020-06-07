import matplotlib.pyplot as plt
import numpy as np
import missingno as msno


def replace_to_nan(dataframe):
    return dataframe.replace(-1, np.nan)


# Get Clean dataframes by dropping all the rows which have missing values
def drop_nan_rows(dataframe, print_stats=False):
    clean_df = dataframe.dropna(axis=0, how='any')
    if print_stats:
        print('Original Length=', len(dataframe), '\tCleaned Length=', len(clean_df), '\tMissing Data=', len(dataframe) - len(clean_df))
    return clean_df


# generate the sparsity matrix (figure) for all the dataframes
def generate_sparsity_matrix(dataframe, do_print=False, save=False):
    missing_df = dataframe.columns[dataframe.isnull().any()].tolist()
    ax0 = fig = msno.matrix(dataframe[missing_df], figsize=(20, 5))
    if print:
        plt.show()
    if save:
        plt.savefig('sparsity.png')


def check_data_imbalance(dataframe):
    print(dataframe.groupby('target').size())
    false_percent = (dataframe['target'].tolist().count(0) / len(dataframe['target'].tolist())) * 100
    print('(label 0) percentage: ' + str(false_percent) + '%')
    true_percentage = (dataframe['target'].tolist().count(1) / len(dataframe['target'].tolist())) * 100
    print('(label 1) percentage: ' + str(true_percentage) + '%')
    print('-' * 64)


def analyse_dataset(dataframe):
    nan_df = replace_to_nan(dataframe)
    # z = drop_nan_rows(nan_df, True)
    generate_sparsity_matrix(nan_df, True)
    check_data_imbalance(nan_df)
