import time
import algorithms
from data_loader import get_train_df


def perf_test(data, target):
    params = {}
    times = {}

    start = time.perf_counter()
    params["brute"] = algorithms.perform_brute_force(data, target)
    end = time.perf_counter()
    times["brute"] = end - start

    start = time.perf_counter()
    params["muta_0.2"] = algorithms.perform_mutation_evolution(data, target, 0.2)
    end = time.perf_counter()
    times["muta_0.2"] = end - start

    start = time.perf_counter()
    params["muta_0.4"] = algorithms.perform_mutation_evolution(data, target, 0.4)
    end = time.perf_counter()
    times["muta_0.4"] = end - start

    start = time.perf_counter()
    params["hill_0.5_4"] = algorithms.perform_hill_climbing(data, target, max_worse=4, ratio=0.5)
    end = time.perf_counter()
    times["hill_0.5_4"] = end - start

    start = time.perf_counter()
    params["hill_0.5_8"] = algorithms.perform_hill_climbing(data, target, max_worse=8, ratio=0.5)
    end = time.perf_counter()
    times["hill_0.5_8"] = end - start

    start = time.perf_counter()
    params["hill_0.75_4"] = algorithms.perform_hill_climbing(data, target, max_worse=4, ratio=0.75)
    end = time.perf_counter()
    times["hill_0.75_4"] = end - start

    start = time.perf_counter()
    params["hill_0.75_8"] = algorithms.perform_hill_climbing(data, target, max_worse=8, ratio=0.75)
    end = time.perf_counter()
    times["hill_0.75_8"] = end - start

    print(params)
    print(times)
    # TODO : eval on test dataset

    return params, times


if __name__ == '__main__':
    # prepare dataset
    driver_data_clean = get_train_df()
    driver_data_clean = driver_data_clean.replace_to_nan(driver_data_clean)

    driver_data_clean = driver_data_clean.drop('ps_car_03_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_car_05_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_reg_03', axis=1)
    driver_data_clean = driver_data_clean.dropna(axis=0, how='any')

    # TODO : count
    # data and target
    driver_target = driver_data_clean['target']
    driver_data = driver_data_clean.drop(['id', 'target'], axis=1)

    driver_params, driver_times = perf_test(driver_data, driver_target)
