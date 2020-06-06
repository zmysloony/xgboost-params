import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


# def smote_oversample(features, labels, seed, neighbours=15):
#     smote = SMOTE(random_state=seed, k_neighbors=neighbours)
#     return smote.fit_resample(features, labels)


def train_score(data, target, params, rounds=1, k_fold_ratio=3, rand=3228):
    k_fold = KFold(n_splits=k_fold_ratio, random_state=rand, shuffle=True)

    recalls = []
    data_arr = np.array(data)
    target_arr = np.array(target)

    for train_index, test_index in k_fold.split(data_arr):
        train_X, valid_X = data_arr[train_index], data_arr[test_index]
        train_y, valid_y = target_arr[train_index], target_arr[test_index]

        d_train = xgb.DMatrix(train_X, label=train_y)
        d_valid = xgb.DMatrix(valid_X, label=valid_y)

        k_model = xgb.train(params, d_train, rounds)

        k_predict = k_model.predict(d_valid)

        k_recall = roc_auc_score(valid_y, k_predict)
        recalls.append(k_recall)
    return np.average(recalls)


# data - dataset, target - predictions,
# params - xgboost parameters, rounds - xgboost rounds,
# k_fold_ratio - no of k-folds, rand - seed for k_fold and xgboost
# optimized parameters = params & rounds
def train(data, target, params, rounds=1, k_fold_ratio=3, rand=3228):
    final_train = xgb.DMatrix(data, label=target)
    model = xgb.train(params, final_train, rounds)

    return model, train_score(data, target, params, rounds, k_fold_ratio, rand)