import os
import pandas as pd
from enum import Enum
import numpy as np
from sklearn.svm import SVR
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

os.chdir(Path(__file__).parents[2])

class PreProcessing(Enum):
    MEAN = 1
    MEDIAN = 2
    MIXED = 3

def plotBestParams(X_train, y_train):
    svr = SVR()
    param_grid = {
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree' : [2, 3, 4],
        'epsilon' : list(np.arange(0, 0.2, 0.0125)),
    }

    gs = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5)
    gs.fit(X_train, y_train)

    degree = gs.best_params_['degree']
    epsilon = gs.best_params_['epsilon']
    kernel = gs.best_params_['kernel']

    return degree, epsilon, kernel

if __name__ == "__main__":
    usedPreprocessing = PreProcessing.MIXED
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]

    y_pred = []
    y_test = []

    if usedPreprocessing == PreProcessing.MEAN:
        train_mean = pd.read_csv("data/05_numeric/Universities_train_mean.csv")
        test_mean = pd.read_csv("data/05_numeric/Universities_test_mean.csv")

        y_test.clear()
        y_pred.clear()
        # separate data frame to X and y
        X_train_mean = train_mean.drop(columns=target_columns).to_numpy()
        y_train_mean = train_mean[target_columns]

        X_test_mean = test_mean.drop(columns=target_columns).to_numpy()
        y_test_mean = test_mean[target_columns]

        for target in target_columns:
            degree, epsilon, kernel = plotBestParams(X_train_mean, y_train_mean[target].to_numpy())
            svr_mean = SVR(kernel=kernel, degree=degree, epsilon=epsilon)
            svr_mean.fit(X_train_mean, y_train_mean[target].to_numpy())
            y_pred.append(svr_mean.predict(X_test_mean))
            y_test.append(y_test_mean[target].to_numpy())

    if usedPreprocessing == PreProcessing.MEDIAN:
        train_median = pd.read_csv("data/05_numeric/Universities_train_median.csv")
        test_median = pd.read_csv("data/05_numeric/Universities_test_median.csv")

        y_test.clear()
        y_pred.clear()
        # separate data frame to X and y
        X_train_median = train_median.drop(columns=target_columns).to_numpy()
        y_train_median = train_median[target_columns]

        X_test_median = test_median.drop(columns=target_columns).to_numpy()
        y_test_median = test_median[target_columns]

        for target in target_columns:
            degree, epsilon, kernel = plotBestParams(X_train_median, y_train_median[target].to_numpy())
            svr_median = SVR(kernel=kernel, degree=degree, epsilon=epsilon)
            svr_median.fit(X_train_median, y_train_median[target].to_numpy())
            y_pred.append(svr_median.predict(X_test_median))
            y_test.append(y_test_median[target].to_numpy())

    if usedPreprocessing == PreProcessing.MIXED:
        train_mixed = pd.read_csv("data/05_numeric/Universities_train_mixed.csv")
        test_mixed = pd.read_csv("data/05_numeric/Universities_test_mixed.csv")

        y_test.clear()
        y_pred.clear()
        # separate data frame to X and y
        X_train_mixed = train_mixed.drop(columns=target_columns).to_numpy()
        y_train_mixed = train_mixed[target_columns]

        X_test_mixed = test_mixed.drop(columns=target_columns).to_numpy()
        y_test_mixed = test_mixed[target_columns]

        for target in target_columns:
            degree, epsilon, kernel = plotBestParams(X_train_mixed, y_train_mixed[target].to_numpy())
            svr_mixed = SVR(kernel=kernel, degree=degree, epsilon=epsilon)
            svr_mixed.fit(X_train_mixed, y_train_mixed[target].to_numpy())
            y_pred.append(svr_mixed.predict(X_test_mixed))
            y_test.append(y_test_mixed[target].to_numpy())

    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[0],
                                    "predicted PG_average_fees_(in_pounds)": y_pred[1],
                                    "truth UG_average_fees_(in_pounds)": y_test[0],
                                    "truth PG_average_fees_(in_pounds)": y_test[1]})

    predicted_truth.to_csv("results/SVR_predicted_truth.csv", index=False)
