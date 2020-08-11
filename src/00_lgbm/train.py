import glob
import os
from pprint import pprint

import numpy as np
import pandas as pd
from optuna.integration import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

root = os.path.join(
    os.environ["HOME"], "Workspace/learning/signate/SIGNATE_Student_Cup_2020"
)


def load_data():
    train = pd.read_csv(os.path.join(root, "processed_data", "00_train.csv"))
    X, y = train.iloc[:, :-2], train["jobflag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=123, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train(params, X_train, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    lgb_train = lgb.Dataset(X_train, y_train)

    tuner_cv = lgb.LightGBMTunerCV(
        params=params,
        train_set=lgb_train,
        folds=folds,
        feval=lgb_f1_score,
    )

    tuner_cv.run()

    print(f"Best score: {tuner_cv.best_score}")
    print("Best params:")
    pprint(tuner_cv.best_params)

    return tuner_cv.best_params


def test(best_params, X_train, X_test, y_train, y_test):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test)

    mdl = lgb.train(best_params, lgb_train, valid_sets=lgb_test, feval=lgb_f1_score)
    pred = mdl.predict(X_test).argmax(axis=1)
    
    print("*** test score ***")
    print(f1_score(y_test, pred, average="macro"))


def lgb_f1_score(preds, data):
    y_true = data.get_label()
    preds = preds.reshape(4, len(preds) // 4)
    y_pred = np.argmax(preds, axis=0)
    score = f1_score(y_true, y_pred, average="macro")
    return "custom", score, True


def main():
    params = {
        # fixed
        "task": "train",
        "objective": "multiclass",
        "num_classes": 4,
        "seed": 0,
        "boosting_type": "gbdt",
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_boost_round": 1000,
        "metric": "custom",
        "verbosity": -1,
        "verbose_eval": 50,
    }

    X_train, X_test, y_train, y_test = load_data()
    best_params = train(params, X_train, y_train)
    test(best_params, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
