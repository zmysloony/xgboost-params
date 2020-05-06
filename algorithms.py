from collections.abc import Mapping
from itertools import product

from model import train as trainXGBmodel


class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter."""

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.

        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params


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


def brute_force_approach(data, target, params_dict):
    optimal_params = {}
    # sklearn GridSearchCV must use different api than model.train approach
    #
    # another option is to use sklearn-like API in model.train
    #   [but what with impyute and oversample ? we want to modify only learning dataset]
    for it in ParameterGrid(params_dict):
        for n_trees in range(1, 12):
            print("Params :\n", it, "\ntrees : ", n_trees)  # TODO search and compare
            optimal_params = it
            trainXGBmodel(data, target, it, n_trees, 2)

    # return also model and metrics?
    return optimal_params
