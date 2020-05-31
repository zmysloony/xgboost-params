from typing import List

from model import train as trainXGB
import copy


class ParamData:
    name = ''
    min = 0
    max = 0
    step = 0

    def __init__(self, name, span, step):
        self.name = name
        self.min = span[0]
        self.max = span[1]
        self.step = step

    def __str__(self):
        ret = "name: " + self.name + ", range: [" + str(self.min) + ", " + str(self.max) + "], step: " + str(self.step)
        return ret


class Param:
    param_data = None
    value = None

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
    params: List[Param]
    score = 0

    def __init__(self, params):
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
        model, auc_roc = trainXGB(data, target, self.extract_params(), self.n_estimators().value)
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


class HistoricalSet:
    extracted_params = {}

    def __init__(self, extracted_params):
        self.extracted_params = extracted_params

    def compare(self, other_params):
        return other_params == self.extracted_params


class ParamSetsHistory:
    historical_sets = []

    def add_set(self, param_set: ParamSet):
        self.historical_sets.append(HistoricalSet(param_set.extract_params()))

    def has_set(self, param_set: ParamSet):
        for hs in self.historical_sets:
            if hs.compare(param_set.extract_params()):
                return True
        return False


def perform_hill_climbing(data, target):
    starting_params = [Param(ParamData('max_depth', [1, 24], 1)),
                       Param(ParamData('eta', [0.1, 0.4], 0.1))]

    history = ParamSetsHistory()
    paramset = ParamSet(starting_params)
    history.add_set(paramset)
    print(paramset)
    for p in paramset.params:
        print(p.param_data)

    for n in paramset.generate_neighbors(history):
        history.add_set(n)
        print(n, "\nnext:\n")
        for z in n.generate_neighbors(history):
            history.add_set(z)
            print(z)
        print("\n\n\n")

    params_dict = {'max_depth': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 24],
                   'eta': [0.1, 0.2, 0.3, 0.4],
                   'gamma': [0, 1, 2, 3],
                   'subsample': [0.6, 0.8, 1],
                   'colsample_bytree': [0.6, 0.8, 1],
                   'max_delta_step': [0, 1, 2],
                   'eval_metric': ['auc'],
                   'tree_method': ['hist'],
                   'silent': [1],
                   'nthread': [0],
                   'n_estimators': [1, 10, 25, 30]
                   }