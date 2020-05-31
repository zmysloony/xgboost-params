from typing import List

from model import train as trainxgb
import copy


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


class ParamSet:
    def __init__(self, params: List[Param]):
        self.score = None
        self.params = params

    def __str__(self):
        ret = "Score: " + str(self.score) + " Params:"
        for p in self.params:
            ret += "\n\t" + p.param_data.name + " = " + str(p.value)
        return ret

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
        return ls

    def train(self, data, target):
        model, auc_roc = trainxgb(data, target, self.extract_params(), self.n_estimators().value)
        self.score = auc_roc
        return model

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

    def max_combinations(self):
        total = 1
        for p in self.params:
            total *= len(range(p.param_data.min, p.param_data.max, p.param_data.step)) + 1
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
def perform_hill_climbing(data, target, max_worse=8):
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
    i = 0
    while current_set is not None and worse_count < max_worse:   # equals none when all neighbors expanded
        try:
            i += 1
            print("Set " + str(i) + "/" + str(total_combinations))
            current_set.train(data, target)
            history.add_set(current_set)
            history.mark_expanded(current_set)

            neighbors = current_set.generate_neighbors(history)
            best_nb = best_neighbor(neighbors, history, data, target)
            if best_nb is not None and best_nb.score >= current_set.score:   # climb  up
                print("Climbing from " + str(current_set.score) + " to " + str(best_nb.score))
                current_set = best_nb
                if best_nb.score > best_set.score:
                    worse_count = 0
                    best_set = best_nb
                else:
                    worse_count += 1
            else:   # go down to one of the visited ones
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
    else:
        print("Count of iterations with worse than current best result exceeded (" + str(max_worse) + ") - returning best result")
    return best_set

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
