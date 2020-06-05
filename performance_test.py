import time
import algorithms
import numpy as np
from data_analysis import replace_to_nan
from data_loader import get_train_df


def perf_test(data, target):
    params = {}
    times = {}

    # start = time.perf_counter()
    # params["brute"] = algorithms.perform_brute_force(data, target)
    # end = time.perf_counter()
    # times["brute"] = end - start
    # print('end brute')

    for part in np.arange(0.2, 0.65, 0.15):
        start = time.perf_counter()
        params['muta_' + str(part)] = algorithms.perform_mutation_evolution(data, target, part).extract_params()
        end = time.perf_counter()
        times['muta_' + str(part)] = end - start
        print('end muta_' + str(part))

    # for part in np.arange(0.2, 0.65, 0.15):
    #     for worse in [8, 16, 24]:
    #         start = time.perf_counter()
    #         params["hill_" + str(part) + '_' + str(worse)] = algorithms.perform_hill_climbing(data, target,
    #                                                                                           max_worse=worse,
    #                                                                                           ratio=part).extract_params()
    #         end = time.perf_counter()
    #         times["hill_" + str(part) + '_' + str(worse)] = end - start
    #         print('end hill_' + str(part) + '_' + str(worse))

    print(params)
    print(times)
    # TODO : eval on test dataset

    return params, times


if __name__ == '__main__':
    # prepare dataset
    driver_data_clean = get_train_df()
    driver_data_clean = replace_to_nan(driver_data_clean)

    driver_data_clean = driver_data_clean.drop('ps_car_03_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_car_05_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_reg_03', axis=1)
    driver_data_clean = driver_data_clean.dropna(axis=0, how='any')

    driver_data_clean = driver_data_clean.head(100000)
    # TODO : count
    # data and target
    driver_target = driver_data_clean['target']
    driver_data = driver_data_clean.drop(['id', 'target'], axis=1)

    driver_params, driver_times = perf_test(driver_data, driver_target)
