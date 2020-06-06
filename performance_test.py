import time
import algorithms
import numpy as np
from data_analysis import replace_to_nan
from data_loader import get_train_df


def perf_test(data, target):
    params = {}
    times = {}
    scores = {}

    # brute force - evaluating every possible parameter set
    print("Starting brute force algorithm...")
    start = time.perf_counter()
    params["brute"] = algorithms.perform_brute_force(data, target)
    end = time.perf_counter()
    times["brute"] = end - start
    print('brute - best parameter set:\n' + str(params['brute']))

    # mutation algorithm
    print("Starting mutation algorithm...")
    for part in np.arange(0.2, 0.65, 0.15):
        experiment_name = 'muta_' + str(part)

        start = time.perf_counter()
        param_set = algorithms.perform_mutation_evolution(data, target, part)
        end = time.perf_counter()

        params[experiment_name] = param_set.extract_params()
        scores[experiment_name] = param_set.score
        times[experiment_name] = end - start
        print(experiment_name, " - best parameter set:\n" + str(param_set))

    # hill climbing algorithm
    print("Starting hill climbing algorithm...")
    for part in np.arange(0.2, 0.65, 0.15):
        for worse in [8, 16, 24]:
            experiment_name = 'hill_' + str(part) + '_' + str(worse)

            start = time.perf_counter()
            param_set = algorithms.perform_hill_climbing(data, target, max_worse=worse, ratio=part)
            end = time.perf_counter()

            params[experiment_name] = param_set.extract_params()
            scores[experiment_name] = param_set.score
            times[experiment_name] = end - start
            print(experiment_name, " - best parameter set:\n" + str(param_set))

    # TODO : eval on test dataset

    return params, scores, times


if __name__ == '__main__':
    # prepare dataset
    driver_data_clean = get_train_df()
    driver_data_clean = replace_to_nan(driver_data_clean)

    # TODO : decide if we really should remove these attribs
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
