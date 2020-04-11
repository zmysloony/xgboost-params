import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


def train(data, target, test_ratio=0.33):
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
    print("Accuracy: %.2f%%" % (accuracy*100.0))
    print(precision_score(y_test, best_preds, average='macro'))
    return model
