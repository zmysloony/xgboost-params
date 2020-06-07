import argparse
import signal
import time
import algorithms
import numpy as np

import data_analysis
from alg_utils import str_extracted_params
from data_analysis import replace_to_nan
from data_loader import get_train_df, check_data_files
from model import set_use_gpu


def timeout_handler(signum, frame):
    raise KeyboardInterrupt("timer")


def perf_test(data, target, t_args: {}):
    signal.signal(signal.SIGALRM, timeout_handler)
    params = {}
    times = {}
    scores = {}
    iterations = {}
    max_iterations = {}

    if t_args['bruteforce'] is False and t_args['mutation'] is False and t_args['hillclimb'] is False:
        t_args['bruteforce'] = True
        t_args['mutation'] = True
        t_args['hillclimb'] = True

    # brute force - evaluating every possible parameter set
    if t_args['bruteforce']:
        experiment_name = 'brute'
        print("Starting brute force algorithm...")
        if t_args['timeout']:
            signal.alarm(t_args['timeout'])
        params[experiment_name], scores[experiment_name], done_iter, max_iter, a_times = algorithms.perform_brute_force(data, target)
        iterations[experiment_name] = done_iter
        times[experiment_name] = a_times
        max_iterations[experiment_name] = max_iter
        if params[experiment_name] is not None and scores[experiment_name] is not None:
            ret = str_extracted_params(params[experiment_name], scores[experiment_name][-1])
            print('[brute force] Explored', done_iter, 'out of', max_iter, 'options. Best parameter set:\n' + ret)

    # mutation algorithm
    if t_args['mutation']:
        print("Starting mutation algorithm...")
        # for part in np.arange(0.2, 0.65, 0.15):
        for part in [0.05]:
            experiment_name = 'muta_' + str(part)

            if t_args['timeout']:
                signal.alarm(t_args['timeout'])
            param_set, done_iter, max_iter, a_scores, a_times = algorithms.perform_mutation_evolution(data, target, part)

            params[experiment_name] = param_set.extract_params()
            scores[experiment_name] = a_scores
            times[experiment_name] = a_times
            iterations[experiment_name] = done_iter
            max_iterations[experiment_name] = max_iter
            print('['+experiment_name+'] Explored', done_iter, 'out of', max_iter, 'options. Best parameter set:\n'
                  + str(param_set))

    # hill climbing algorithm
    if t_args['hillclimb']:
        print("Starting hill climbing algorithm...")
        #for part in np.arange(0.2, 0.65, 0.15):
            #for worse in [8, 16, 24]:
        for part in [0.05]:
            for worse in [6]:
                experiment_name = 'hill_' + str(part) + '_' + str(worse)

                start = time.perf_counter()
                if t_args['timeout']:
                    signal.alarm(t_args['timeout'])
                param_set, done_iter, max_iter, a_scores, a_times = algorithms.perform_hill_climbing(data, target, max_worse=worse, ratio=part)
                end = time.perf_counter()

                params[experiment_name] = param_set.extract_params()
                scores[experiment_name] = a_scores
                times[experiment_name] = a_times
                iterations[experiment_name] = done_iter
                max_iterations[experiment_name] = max_iter
                print('[' + experiment_name + '] Explored', done_iter, 'out of', max_iter,
                      'options. Best parameter set:\n' + str(param_set))

    return params, scores, times, iterations, max_iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost hyperparameter adjustment using heuristics')
    parser.add_argument('-a', '--analysis', help='Run data analysis and output results.', required=False, action='store_true')
    parser.add_argument('-t', '--timeout', help='Set timeout for each algorithm when testing.', required=False, default=300)
    parser.add_argument('-l', '--length', help='Select how many record of data to extract for learning and analysis.', required=False, default=595212)
    parser.add_argument('-b', '--bruteforce', help='Test brute force (parameter grid) algorithm.', required=False, action='store_true')
    parser.add_argument('-m', '--mutation', help='Test mutation algorithm.', required=False, action='store_true')
    parser.add_argument('-c', '--climbhill', help='Test hill climbing algorithm.', required=False, action='store_true')
    parser.add_argument('-g', '--gpu', help='Use GPU acceleration (only works with nvidia CUDA).', required=False, action='store_true')
    args = parser.parse_args()

    # prepare dataset
    driver_data = get_train_df()
    driver_data_clean = replace_to_nan(driver_data)

    # global print_debug
    if args.gpu:
        set_use_gpu()
    if args.analysis:
        print('Starting dataset analysis...')
        if not check_data_files():
            exit(1)
        data_analysis.analyse_dataset(driver_data)

    if int(args.timeout) < 5:
        args.timeout = 5
    perftest_args = {'bruteforce': args.bruteforce, 'mutation': args.mutation, 'hillclimb': args.climbhill, 'timeout': int(args.timeout)}

    driver_data_clean = driver_data_clean.drop('ps_car_03_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_car_05_cat', axis=1)
    driver_data_clean = driver_data_clean.drop('ps_reg_03', axis=1)
    driver_data_clean = driver_data_clean.dropna(axis=0, how='any')

    driver_data_clean = driver_data_clean.head(int(args.length))

    # data and target
    driver_target = driver_data_clean['target']
    driver_data = driver_data_clean.drop(['id', 'target'], axis=1)

    f_params, f_scores, f_times, f_iter, f_max_iter = perf_test(driver_data, driver_target, perftest_args)
    # structure: name, best_score, parameters, done_iterations, max_iterations, {{running_time, score}, ...}
    exp = list('')
    for k, v in f_params.items():
        exp += '{' + k + ', ' + str(f_scores[k][-1]) + ', ' + str(v) + ', ' + str(f_iter[k]) + ', ' + str(f_max_iter[k]) + ', {'
        for s, t in zip(f_scores[k], f_times[k]):
            exp += '{' + str(s) + ', ' + str(t) + '},'
        if exp[-1] == ',':
            exp[-1] = '}'
        else:
            exp += '}'
        exp += "},"
    if exp[-1] == ',':
        exp[-1] = '}'
    else:
        exp += '}'
    opened = open("results"+time.strftime("%Y%m%d-%H%M%S")+".txt", "w")
    opened.write(''.join(exp))
    opened.close()

