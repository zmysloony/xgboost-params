import time

from model import train_score as trainxgb
from alg_utils import Param, ParamData, ParamSet, ParamSetsHistory, ParameterGrid, str_extracted_params
import numpy as np


print_debug = False


def dprint(*printargs):
    if print_debug:
        print(*printargs)


def brute_force_approach(data, target, params_dict):
    #TODO : RANDOMIZE STARTING PARAMS!!!
    start_timer = time.perf_counter()
    optimal_params = None
    optimal_score = None
    times = []
    scores = []
    grid = ParameterGrid(params_dict)
    ntree_grid = [50, 100, 150, 200, 250, 300]
    max_iter = 1
    for k, v in params_dict.items():
        max_iter *= len(v)
    max_iter *= len(ntree_grid)
    i = 0
    try:
        for it in grid:
            for n_trees in ntree_grid:
                i += 1
                dprint('train')
                score = trainxgb(data, target, it, n_trees)
                if optimal_score is None or optimal_params is None:
                    scores.append(score)
                    times.append(time.perf_counter() - start_timer)
                    optimal_score = score
                    optimal_params = {'dict': it, 'n_trees': n_trees}
                else:
                    if optimal_score < score:
                        scores.append(score)
                        times.append(time.perf_counter() - start_timer)
                        optimal_score = score
                        optimal_params = {'dict': it, 'n_trees': n_trees}
    except KeyboardInterrupt as e:
        if len(e.args) == 1 and e.args[0] == 'timer':
            optimal_params['dict']['n_estimators'] = optimal_params['n_trees']
            return optimal_params['dict'], scores, i, max_iter, times
        else:
            if optimal_score is not None and optimal_params is not None:
                print("best found so far :\n")
                optimal_params['dict']['n_estimators'] = optimal_params['n_trees']
                str_extracted_params(optimal_params['dict'], optimal_score)
            exit(1)
    optimal_params['dict']['n_estimators'] = optimal_params['n_trees']
    return optimal_params['dict'], scores, i, max_iter, times


def perform_brute_force(data, target):
    params_dict = {'max_depth': [i for i in range(4, 15)],
                   'eta': [0.1, 0.2, 0.3, 0.4],
                   'gamma': [0, 1, 2, 3],
                   'subsample': [0.6, 0.8, 1],
                   'colsample_bytree': [0.6, 0.8, 1],
                   # 'scale_pos_weigth': [1, 4, 7, 10],
                   'max_delta_step': [0, 1, 2],
                   }

    return brute_force_approach(data, target, params_dict)


def best_neighbor(neighbors, history, data, target, counter: []):
    maxscore = 0
    best = None
    for n in neighbors:
        dprint('trained')
        n.train(data, target)
        history.add_set(n)
        if n.score > maxscore:
            maxscore = n.score
            best = n
        counter[0] += 1
    return best


# max_worse defines how many max. times we can expand a node, that gives worse results
# than the best result - if that count exceeds max_worse -> returns best param set
def perform_hill_climbing(data, target, max_worse=8, ratio=0.7):
    starting_params = [Param(ParamData('n_estimators', [50, 300], 50)),
                       #Param(ParamData('scale_pos_weight', [1, 10], 3)),
                       Param(ParamData('max_depth', [4, 14], 1)),
                       Param(ParamData('eta', [0.1, 0.4], 0.1)),
                       Param(ParamData('gamma', [0, 2], 1)),
                       Param(ParamData('subsample', [0.6, 1], 0.2)),
                       Param(ParamData('colsample_bytree', [0.6, 1], 0.2)),
                       Param(ParamData('max_delta_step', [0, 2], 1))]

    history = ParamSetsHistory()

    best_set = ParamSet(starting_params)

    # rand start point
    best_set = best_set.gen_rand(np.random.randint(2147483647))

    start_time = time.perf_counter()
    times = [0]
    scores = [best_set.score]
    current_set = best_set
    worse_count = 0
    total_combinations = best_set.max_combinations()
    max_iter = int(total_combinations * ratio) + 1
    i = [1]
    try:
        while current_set is not None and worse_count < max_worse and max_iter >= i[0]:
            # current_set equals None when all neighbors expanded

            dprint("Set " + str(i[0]) + "/" + str(total_combinations))
            current_set.train(data, target)
            history.add_set(current_set)
            history.mark_expanded(current_set)

            neighbors = current_set.generate_neighbors(history)
            best_nb = best_neighbor(neighbors, history, data, target, i)
            if best_nb is not None and best_nb.score >= current_set.score:  # climb  up
                if best_nb.score > current_set.score:
                    dprint("Climbing from " + str(current_set.score) + " to " + str(best_nb.score))
                current_set = best_nb
                if best_nb.score > best_set.score:
                    scores.append(best_nb.score)
                    times.append(time.perf_counter() - start_time)
                    worse_count = 0
                    best_set = best_nb
                else:
                    worse_count += 1
            else:  # go down to one of the visited ones
                worse_count += 1
                current_set = history.get_best_not_expanded()
                dprint(current_set)

        if current_set is None:
            dprint("Exhausted all options.")
        elif i[0] == max_iter:
            dprint("Reached iteration limit.")
        else:
            dprint("Count of iterations with worse than current best result exceeded (" + str(
                max_worse) + ") - returning best result")
    except KeyboardInterrupt as e:
        if len(e.args) == 1 and e.args[0] == 'timer':
            return best_set, i[0], max_iter, scores, times
        else:
            print("best found so far :\n", current_set)
            exit(1)
    return best_set, i[0], max_iter, scores, times


# ratio - how many combinations to check
def perform_mutation_evolution(data, target, ratio=0.4, seed=42):
    starting_params = [Param(ParamData('n_estimators', [50, 300], 50)),
                       #Param(ParamData('scale_pos_weight', [1, 100], 33)),
                       Param(ParamData('max_depth', [4, 14], 1)),
                       Param(ParamData('eta', [0.1, 0.4], 0.1)),
                       Param(ParamData('gamma', [0, 2], 1)),
                       Param(ParamData('subsample', [0.6, 1], 0.2)),
                       Param(ParamData('colsample_bytree', [0.6, 1], 0.2)),
                       Param(ParamData('max_delta_step', [0, 2], 1))]

    history = ParamSetsHistory()

    current_set = ParamSet(starting_params)
    total_combinations = current_set.max_combinations()
    max_iter = int(total_combinations * ratio) + 1
    iteration = 1
    start_time = time.perf_counter()
    times = []
    scores = []
    try:
        np.random.seed(seed)

        # rand start point
        current_set = current_set.gen_rand(np.random.randint(2147483647))

        # init and first train
        current_set.train(data, target)
        times.append(time.perf_counter() - start_time)
        scores.append(current_set.score)
        history.add_set(current_set)
        mutants = current_set.gen_unvisited_mutations(history)

        while iteration <= max_iter:  # equals none when all mutants expanded

            # if len(mutants) == 0:
            #     mutants = current_set.generate_neighbors_wider(history)
            #     if len(mutants) == 0:
            #         break
            #     # TODO remove after fix generating mutants
            if len(mutants) == 0:
                break

            mutant = mutants[np.random.randint(0, len(mutants))]
            mutant.train(data, target)
            history.add_set(mutant)

            if mutant.score >= current_set.score:
                if mutant.score > current_set.score:
                    dprint("New best, change from " + str(current_set.score) + " to " + str(mutant.score))
                    times.append(time.perf_counter() - start_time)
                    scores.append(current_set.score)
                current_set = mutant

            iteration += 1
            # mutants = TODO : current_set.generate_neighbors(history) - replace with method generating mutants -
            #                    with possibly wider range than 1 step - always at least n
            #              NOTE : here non of sets are marked 'expanded'
            mutants = current_set.gen_unvisited_mutations(history)
        if iteration == max_iter:
            dprint("No more iterations left.")
        else:
            dprint("No more mutants, after " + str(iteration) + '/' + str(max_iter) + " iterations.")
    except KeyboardInterrupt as e:
        if len(e.args) == 1 and e.args[0] == 'timer':
            return current_set, iteration, max_iter, scores, times
        else:
            print("best found so far :\n", current_set)
            exit(1)

    dprint("best found:\n", current_set.score)
    return current_set, iteration, max_iter, scores, times
