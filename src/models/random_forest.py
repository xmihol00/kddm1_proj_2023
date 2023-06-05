#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from pathlib import Path
import os, sys
import random as rn

os.chdir(Path(__file__).parents[2])
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.seed import RANDOM_SEED

np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

# Create dir 'plots/'
os.makedirs("plots", exist_ok=True)


def performGridSearch(X_train, y_train, rf, param_grid, create_Plot = False, plotSuffix = ''):
    # Grid Search
    # https://scikit-learn.org/stable/modules/generated/sklearn.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=5)
    gs.fit(X_train, y_train)
    print('best params:', gs.best_params_)

    # Elbow Graph
    # optimizes accuracy without adding unnecessary performance complexity
    scores = gs.cv_results_['mean_test_score']
    # print("scores: \n{}".format(scores))

    if create_Plot:
        plt.figure()
        plt.plot(range(len(scores)), scores)
        plt.grid()
        plt.xlabel('params')
        plt.ylabel('neg_mean_squared_error')
        plt.savefig('plots/rand_forest_elbow_graph_{}.png'.format(plotSuffix))
        # plt.show()
    return gs.best_params_


def plotBestParams(X_train: np.ndarray, y_train: np.ndarray):

    # #############################################
    # # param: max_features, best params: {'max_features': 21}
    # rf = RandomForestRegressor(random_state=RANDOM_SEED)
    # param_grid = {
    #     'max_features': list(range(1, X_train.shape[1], 1))
    # }
    # print('grid: \n', param_grid)
    # best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'max_features')
    
    # ##############################################
    # # param: n_estimators
    # rf = RandomForestRegressor(random_state=RANDOM_SEED)
    # param_grid = {
    #     'n_estimators': list(range(1, 101, 1)),
    # }
    # print('grid: \n', param_grid)
    # best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'n_estimators')
    
    
    ##############################################
    # param: n_estimators
    # # best params: {'max_features': 17, 'n_estimators': 18}
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    param_grid = {
        'max_features': (10, 17),
        'n_estimators': list(range(1, 101, 1)),
    }
    print('grid: \n', param_grid)
    best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'mix1')
    
    # #############################################
    # # param: mix
    # # best params: {'max_features': 17, 'n_estimators': 38}
    # rf = RandomForestRegressor(random_state=RANDOM_SEED)
    # param_grid = {
    #     'max_features': list(range(10, 17, 1)),
    #     'n_estimators': list(range(35, 60, 1)),
    # }
    # print('grid: \n', param_grid)
    # best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'mix2')

    return best_params

def print_index_of_origin_column(ft_imp, n):
    columns_origin = pd.read_csv('data/Universities.csv').columns
    feature_column_idx = []
    for feature in ft_imp[:n].index:
        feature_first_word = str(feature).split('_')[0]
        for i, col in enumerate(columns_origin):
            if str(col).startswith(feature_first_word):
                feature_column_idx.append(i)
                continue
    
    _, idx = np.unique(feature_column_idx, return_index=True)
    
    print()
    print('column index of   origin by relevance:        ', feature_column_idx)
    print('column index of   origin by relevance unique: ', list(np.array(feature_column_idx)[np.sort(idx)]))
    print('column index of   origin by order:            ', sorted(feature_column_idx))
    print('column index of   origin by order     unique: ', sorted(set(feature_column_idx)))


def printPerformance(rf: RandomForestRegressor, X_test, y_test, columns: list):
    # print('random forest accuracy:                   ', rf.score(X_test, y_test))
    # print('prediction:                               ', rf.predict(X_test[0].reshape(1, -1)))
    # print('true value:                               ', y_test[0])
    
    
    n = rf.max_features
    ft_imp = pd.Series(rf.feature_importances_, index=X_columns).sort_values(ascending=False)
    feature_column_idx = [list(columns).index(feature) if feature in columns else None for feature in ft_imp[:n].index]
    print(pd.DataFrame({
        'pos':range(1, n+1),
        'feature importance':ft_imp.head(n).round(3),
        'column_index': feature_column_idx
        }))
    
    print('column index of features by relevance: ', feature_column_idx[:n])
    print('column index of features by order:     ', sorted(feature_column_idx[:n]))
    print_index_of_origin_column(ft_imp, n)
    
    y_pred = rf.predict(X_test)
    y_true = y_test
    print()
    print('Accuracy:                                {:16.3f}'.format(rf.score(X_test, y_test)))
    print()
    print('Mean Squared Error (MSE):                {:16.2f}'.format(metrics.mean_squared_error(y_true, y_pred)))
    print('Root Mean Squared Error (RMSE):          {:16.2f}'.format(metrics.mean_squared_error(y_true, y_pred, squared=False)))
    print('Mean Absolute Error (MAE):               {:16.2f}'.format(metrics.mean_absolute_error(y_true, y_pred)))
    print('R^2:                                     {:16.3f}'.format(metrics.r2_score(y_true, y_pred)))
    # print()
    # print('Mean Absolute Percentage Error (MAPE):   {:16.3f}'.format(metrics.mean_absolute_percentage_error(y_true, y_pred)))
    # print('Accuracy (1 - MAPE):                     {:16.3f}'.format(1 - metrics.mean_absolute_percentage_error(y_true, y_pred)))
    # print('Explained Variance Score:                {:16.3f}'.format(metrics.explained_variance_score(y_true, y_pred)))
    # print('Median Absolute Error:                   {:16.3f}'.format(metrics.median_absolute_error(y_true, y_pred)))


if __name__ == "__main__":
    train_mean = pd.read_csv('data/05_numeric/Universities_train_mean.csv')
    train_median = pd.read_csv('data/05_numeric/Universities_train_median.csv')

    test_mean = pd.read_csv('data/05_numeric/Universities_test_mean.csv')
    test_median = pd.read_csv('data/05_numeric/Universities_test_median.csv')

    #############################################
    # separate data frame to X and y
    y_columns = ['UG_average_fees_(in_pounds)', 'PG_average_fees_(in_pounds)']
    X_columns = train_median.drop(columns=y_columns).columns
    columns = train_median.columns
    X_train = train_median.drop(columns=y_columns).to_numpy()
    y_train = train_median[y_columns].to_numpy()

    X_test = test_median.drop(columns=y_columns).to_numpy()
    y_test = test_median[y_columns].to_numpy()
    results = []

    #############################################
    # best params: {'max_features': 8}
    # best params: {'n_estimators': 29}
    # best params: {'max_features': 17, 'n_estimators': 37}
    # plotBestParams(X_train, y_train)
    
    #############################################
    rf = RandomForestRegressor(max_features=10, n_estimators=80)
    rf.fit(X_train, y_train)

    #############################################
    y_pred = rf.predict(X_test)
    printPerformance(rf, X_test, y_test, columns)


    #############################################
    # crate a data frame with the results of the selected model
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1],
                                    "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})
    
    # print(predicted_truth)
    predicted_truth.to_csv("results/RF_predicted_truth.csv", index=False)
