from model import train_score as trainxgb
from alg_utils import Param, ParamData, ParamSet, ParamSetsHistory, ParameterGrid
from numpy.random import MT19937, Generator


def brute_force_approach(data, target, params_dict):
    optimal_params = None
    optimal_score = None
    for it in ParameterGrid(params_dict):
        for n_trees in [50, 100, 150, 200, 250]:
            model, score = trainxgb(data, target, it, n_trees, 2)
            if optimal_score is None or optimal_params is None:
                optimal_score = score
                optimal_params = {'dict': it, 'n_trees': n_trees}
            else:
                if optimal_score < score:
                    optimal_score = score
                    optimal_params = {'dict': it, 'n_trees': n_trees}

    return optimal_params


def perform_brute_force(data, target):
    params_dict = {'max_depth': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24],
                   'eta': [0.1, 0.2, 0.3, 0.4],
                   'gamma': [0, 1, 2, 3],
                   'subsample': [0.6, 0.8, 1],
                   'colsample_bytree': [0.6, 0.8, 1],
                   'max_delta_step': [0, 1, 2],
                   'eval_metric': ['auc'],
                   'tree_method': ['hist'],
                   'silent': [1],
                   'nthread': [0]
                   }

    return brute_force_approach(data, target, params_dict)


def best_neighbor(neighbors, history, data, target):
    maxscore = 0
    best = None
    for n in neighbors:
        n.train(data, target)
        history.add_set(n)
        if n.score > maxscore:
            maxscore = n.score
            best = n
    return best


# max_worse defines how many max. times we can expand a node, that gives worse results
# than the best result - if that count exceeds max_worse -> returns best param set
def perform_hill_climbing(data, target, max_worse=8, ratio=0.7):
    starting_params = [Param(ParamData('n_estimators', [1, 200], 10)),
                       Param(ParamData('max_depth', [1, 24], 1)),
                       # Param(ParamData('eta', [0.1, 0.4], 0.1)),
                       # Param(ParamData('gamma', [0, 3], 1)),
                       # Param(ParamData('subsample', [0.6, 1], 0.2)),
                       # Param(ParamData('colsample-bytree', [0.6, 1], 0.2)),
                       # Param(ParamData('max_delta_step', [0, 2], 1)),
                       Param(ParamData('silent', [1, 1], 1)),
                       Param(ParamData('nthread', [0, 0], 1))]

    history = ParamSetsHistory()

    best_set = ParamSet(starting_params)
    current_set = best_set
    worse_count = 0
    total_combinations = best_set.max_combinations()
    max_iter = total_combinations * ratio
    i = 0
    while current_set is not None and worse_count < max_worse and max_iter <= i:  # equals none when all neighbors expanded
        try:
            i += 1
            print("Set " + str(i) + "/" + str(total_combinations))
            current_set.train(data, target)
            history.add_set(current_set)
            history.mark_expanded(current_set)

            neighbors = current_set.generate_neighbors(history)
            best_nb = best_neighbor(neighbors, history, data, target)
            if best_nb is not None and best_nb.score >= current_set.score:  # climb  up
                print("Climbing from " + str(current_set.score) + " to " + str(best_nb.score))
                current_set = best_nb
                if best_nb.score > best_set.score:
                    worse_count = 0
                    best_set = best_nb
                else:
                    worse_count += 1
            else:  # go down to one of the visited ones
                worse_count += 1
                current_set = history.get_best_not_expanded()
                print("rollbackstart")
                print(current_set)
                print("rollbackdone")
        except KeyboardInterrupt:
            print("best set:\n", best_set)
            print("current set:\n", current_set)
            return best_set
    if current_set is None:
        print("Exhausted all options.")
    elif i == max_iter:
        print("Reached iteration limit.")
    else:
        print("Count of iterations with worse than current best result exceeded (" + str(
            max_worse) + ") - returning best result")
    return best_set


# ratio - how many combinations to check
def perform_mutation_evolution(data, target, ratio=0.4, seed=42):
    starting_params = [Param(ParamData('n_estimators', [1, 200], 10)),
                       Param(ParamData('max_depth', [1, 24], 1)),
                       # Param(ParamData('eta', [0.1, 0.4], 0.1)),
                       # Param(ParamData('gamma', [0, 3], 1)),
                       # Param(ParamData('subsample', [0.6, 1], 0.2)),
                       # Param(ParamData('colsample-bytree', [0.6, 1], 0.2)),
                       # Param(ParamData('max_delta_step', [0, 2], 1)),
                       Param(ParamData('silent', [1, 1], 1)),
                       Param(ParamData('nthread', [0, 0], 1))]

    history = ParamSetsHistory()

    current_set = ParamSet(starting_params)
    total_combinations = current_set.max_combinations()
    max_iter = total_combinations * ratio
    random_num = Generator(MT19937(seed))

    # init and first train
    current_set.train(data, target)
    history.add_set(current_set)
    history.mark_expanded(current_set)
    # NOTE : generate_neighbors might not be perfect for this algorithm
    #        consider use of dedicated function
    neighbors = current_set.generate_neighbors(history)
    iteration = 1

    while len(neighbors) > 0 and iteration <= max_iter:  # equals none when all neighbors expanded
        try:
            mutant = neighbors[random_num.integers(0, len(neighbors))]
            mutant.train(data, target)
            history.add_set(mutant)

            if mutant.score >= current_set.score:  # climb up
                print("New best, change from " + str(current_set.score) + " to " + str(mutant.score))
                current_set = mutant

            iteration += 1
            neighbors = current_set.generate_neighbors(history)
        except KeyboardInterrupt:
            print("best found:\n", current_set)
            return current_set

    if iteration == max_iter:
        print("No more iterations left.")
    else:
        print("No more neighbors, after " + str(iteration) + " iterations.")

    print("best found:\n", current_set)
    return current_set

    # params_dict = {'max_depth': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24],
    #                'eta': [0.1, 0.2, 0.3, 0.4],
    #                'gamma': [0, 1, 2, 3],
    #                'subsample': [0.6, 0.8, 1],
    #                'colsample_bytree': [0.6, 0.8, 1],
    #                'max_delta_step': [0, 1, 2],
    #                'eval_metric': ['auc'],
    #                'tree_method': ['hist'],
    #                'silent': [1],
    #                'nthread': [0],
    #                'n_estimators': [1, 10, 25, 30]
    #                }
