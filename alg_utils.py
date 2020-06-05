from collections.abc import Mapping
from itertools import product
from typing import List
from model import train_score as trainxgb
import copy
import numpy as np


class ParamData:
    def __init__(self, name, span, step):
        self.name = name
        self.min = span[0]
        self.max = span[1]
        self.step = step

    def __str__(self):
        ret = "name: " + self.name + ", range: [" + str(self.min) + ", " + str(self.max) + "], step: " + str(self.step)
        return ret


class Param:
    def __init__(self, param_data: ParamData, value=None):
        self.param_data = param_data
        if value is None:
            self.value = param_data.min
        else:
            self.value = value

    def can_lower(self):
        if self.value - self.param_data.step < self.param_data.min:
            return False
        return True

    def can_higher(self):
        if self.value + self.param_data.step > self.param_data.max:
            return False
        return True

    def lower(self):
        if self.can_lower():
            return Param(self.param_data, self.value - self.param_data.step)

    def higher(self):
        if self.can_higher():
            return Param(self.param_data, self.value + self.param_data.step)

    def random(self, seed):
        vals = [i for i in
                np.arange(self.param_data.min, self.param_data.max + self.param_data.step, self.param_data.step)]

        if len(vals) == 1:
            return Param(self.param_data, self.value)
        else:
            if isinstance(self.value, int):
                vals.remove(self.value)
            else:
                # floats in python changes value?
                for it in vals:
                    if abs(it - self.value) < 0.0001:
                        vals.remove(it)
            pos = seed % len(vals)
            return Param(self.param_data, vals[pos])

    def gen_all(self):
        vals = [i for i in
                np.arange(self.param_data.min, self.param_data.max + self.param_data.step, self.param_data.step)]

        return [Param(self.param_data, p) for p in vals]


class ParamSet:
    def __init__(self, params: List[Param]):
        self.score = None
        self.params = params

    def __str__(self):
        ret = "Score: " + str(self.score) + " Params:"
        for p in self.params:
            ret += "\n\t" + p.param_data.name + " = " + str(p.value)
        return ret

    # def gen_rand_on_param(self, param_index, seed):
    #     np.random.seed(seed)
    #
    #     new_params = []
    #     i = 0
    #     for p in self.params:
    #         if i == param_index:
    #             new_params.append(self.params[i].random(np.random.randint(2147483647)))
    #         else:
    #             new_params.append(copy.copy(p))
    #         i = i + 1
    #
    #     return ParamSet(new_params)
    #
    # def gen_rand_on_all_params(self, seed, history_sets=None):
    #     np.random.seed(seed)
    #     param_sets = []
    #
    #     it = 0
    #     # try to generate at least one valid
    #     while len(param_sets) == 0 and it < 10:
    #         for i in range(len(self.params)):
    #             new_set = self.gen_rand_on_param(i, np.random.randint(2147483647))
    #             if history_sets is None:
    #                 param_sets.append(new_set)
    #             else:
    #                 if not history_sets.has_set(new_set):
    #                     param_sets.append(new_set)
    #         it = it + 1
    #
    #     return param_sets

    def gen_rand(self, seed):
        np.random.seed(seed)

        new_params = []
        i = 0
        for p in self.params:
            new_params.append(p.random(np.random.randint(2147483647)))

        return ParamSet(new_params)

    def gen_mutants_on_param(self, param):
        new_params = []
        i = 0
        for p in self.params:
            if i != param:
                new_params.append(copy.copy(p))
            i = i + 1

        mutations = self.params[param].gen_all()
        ret = []

        for it in mutations:
            temp = copy.copy(new_params)
            temp.append(it)
            ret.append(ParamSet(temp))

        return ret

    def gen_unvisited_mutations(self, history_sets=None):
        param_sets = []
        new_sets = []
        for i in range(len(self.params)):
            new_sets.extend(self.gen_mutants_on_param(i))

            for s in new_sets:
                if history_sets is None:
                    param_sets.append(s)
                else:
                    if not history_sets.has_set(s):
                        param_sets.append(s)
        return param_sets

    def n_estimators(self):
        p = None
        for p in self.params:
            if p.param_data.name == 'n_estimators':
                break
        return p

    def extract_params(self):
        ls = {}
        for p in self.params:
            ls[p.param_data.name] = p.value
            # if p.name != 'n_estimators':
            #     l[p.name] = p.value
        # TODO
        ls['eval_metric'] = 'auc'
        ls['tree_method'] = 'gpu_hist'
        return ls

    def train(self, data, target):
        auc_roc = trainxgb(data, target, self.extract_params(), self.n_estimators().value)
        self.score = auc_roc
        # return model

    def gen_neighbors_on_parameter(self, param_index):
        new_sets = []

        # add a set with one value lower if possible
        if self.params[param_index].can_lower():
            new_params = []
            i = 0
            for p in self.params:
                if i == param_index:
                    new_params.append(p.lower())
                else:
                    new_params.append(copy.copy(p))
                i = i + 1
            new_sets.append(ParamSet(new_params))

        # add a set with one value higher if possible
        if self.params[param_index].can_higher():
            new_params = []
            i = 0
            for p in self.params:
                if i == param_index:
                    new_params.append(p.higher())
                else:
                    new_params.append(copy.copy(p))
                i = i + 1
            new_sets.append(ParamSet(new_params))

        return new_sets

    def generate_neighbors(self, history_sets=None):
        param_sets = []
        for i in range(len(self.params)):
            new_sets = self.gen_neighbors_on_parameter(i)
            for s in new_sets:
                if history_sets is None:
                    param_sets.append(s)
                else:
                    if not history_sets.has_set(s):
                        param_sets.append(s)
        return param_sets

    def generate_neighbors_wider(self, history_sets=None):
        param_sets = []
        new_sets = []
        for i in range(len(self.params)):
            new_sets = self.gen_neighbors_on_parameter(i)

        for i in new_sets:
            new_wider_sets = i.generate_neighbors(history_sets)
            for s in new_wider_sets:
                if history_sets is None:
                    param_sets.append(s)
                else:
                    if not history_sets.has_set(s):
                        param_sets.append(s)
        return param_sets

    def max_combinations(self):
        total = 1
        for p in self.params:
            # float arnage works for int
            total *= len(np.arange(p.param_data.min, p.param_data.max, p.param_data.step)) + 1
        return total


class HistoricalSet:
    def __init__(self, extracted_params, params: List[Param], score):
        self.params = []
        self.expanded = False
        self.score = score
        self.extracted_params = extracted_params
        for p in params:
            self.params.append(copy.copy(p))

    def compare(self, other_params):
        return other_params == self.extracted_params


class ParamSetsHistory:
    historical_sets = []

    def add_set(self, param_set: ParamSet):
        self.historical_sets.append(HistoricalSet(param_set.extract_params(), param_set.params, param_set.score))

    def mark_expanded(self, param_set: ParamSet):
        for hs in self.historical_sets:
            if hs.compare(param_set.extract_params()):
                hs.expanded = True

    def get_best_not_expanded(self):
        best_score = 0
        best_set = None
        for hs in self.historical_sets:
            if not hs.expanded and hs.score > best_score:
                best_score = hs.score
                best_set = hs
        if best_set is None:
            return None
        bset = ParamSet(best_set.params)
        bset.score = best_score
        return bset

    def has_set(self, param_set: ParamSet):
        for hs in self.historical_sets:
            if hs.compare(param_set.extract_params()):
                return True
        return False


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
