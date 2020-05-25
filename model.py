import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def smote_oversample(features, labels, seed, neighbours=15):
    smote = SMOTE(random_state=seed, k_neighbors=neighbours)
    return smote.fit_resample(features, labels)


# data - dataset, target - predictions,
# params - xgboost parameters, rounds - xgboost rounds,
# k_fold_ratio - no of k-folds, rand - seed for k_fold and xgboost
# optimized parameters = params & rounds
def train(data, target, params, rounds=1, k_fold_ratio=3, rand=3228):
    k_fold = KFold(n_splits=k_fold_ratio, random_state=rand, shuffle=True)

    recalls = []
    data_arr = np.array(data)
    target_arr = np.array(target)

    for train_index, test_index in k_fold.split(data_arr):
        train_X, valid_X = data_arr[train_index], data_arr[test_index]
        train_y, valid_y = target_arr[train_index], target_arr[test_index]

        # oversample training set
        np.where(train_X == np.nan, -1, train_X)
        train_X, train_y = smote_oversample(train_X, train_y, rand)
        np.where(train_X == -1, np.nan, train_X)

        d_train = xgb.DMatrix(train_X, label=train_y)
        d_valid = xgb.DMatrix(valid_X, label=valid_y)

        k_model = xgb.train(params, d_train, rounds)

        k_predict = k_model.predict(d_valid)

        k_recall = roc_auc_score(valid_y, np.asarray([np.argmax(line) for line in k_predict]))
        recalls.append(k_recall)

    final_train = xgb.DMatrix(data, label=target)
    model = xgb.train(params, final_train, rounds)

    k_it = 1
    for rec in recalls:
        print("k = ", k_it, " roc_auc_score = ", rec)
        k_it += 1
    print("avg roc_auc_score = ", np.average(recalls))

    return model, recalls


# data - dataset, target - predictions,
# params - xgboost parameters, rounds - xgboost rounds,
# k_fold_ratio - no of k-folds, rand - seed for k_fold and xgboost
# optimized parameters = params & rounds
def train_undersample(data, target, params, rounds=1, k_fold_ratio=3, rand=3228):
    k_fold = KFold(n_splits=k_fold_ratio, random_state=rand, shuffle=True)

    recalls = []
    data_arr = np.array(data)
    target_arr = np.array(target)

    for train_index, test_index in k_fold.split(data_arr):
        train_X, valid_X = data_arr[train_index], data_arr[test_index]
        train_y, valid_y = target_arr[train_index], target_arr[test_index]

        train_X = np.where(train_X == np.nan, -1, train_X)
        cc = ClusterCentroids(random_state=rand)
        train_X, train_y = cc.fit_resample(train_X, train_y)
        train_X = np.where(train_X == -1, np.nan, train_X)

        d_train = xgb.DMatrix(train_X, label=train_y)
        d_valid = xgb.DMatrix(valid_X, label=valid_y)

        k_model = xgb.train(params, d_train, rounds)

        k_predict = k_model.predict(d_valid)

        k_recall = roc_auc_score(valid_y, np.asarray([np.argmax(line) for line in k_predict]))
        recalls.append(k_recall)

    cc_fin = ClusterCentroids(random_state=rand)
    # TODO undersample
    final_train = xgb.DMatrix(data, label=target)
    model = xgb.train(params, final_train, rounds)

    k_it = 1
    for rec in recalls:
        print("k = ", k_it, " roc_auc_score = ", rec)
        k_it += 1
    print("avg roc_auc_score = ", np.average(recalls))

    return model, recalls


def old_train(data, target, test_ratio=0.33):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_ratio, random_state=0)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3}  # the number of classes that exist in this datset
    num_round = 1  # the number of training iterations
    model = xgb.train(param, dtrain, num_round)
    y_pred = model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in y_pred])
    accuracy = accuracy_score(y_test, best_preds)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.4f%%" % (precision_score(y_test, best_preds, average='macro')) * 100)
    return model
