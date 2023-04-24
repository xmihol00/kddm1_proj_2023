#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os

# Create dir 'plt/'
if not os.path.isdir("plt"):
    os.makedirs("plt")

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
        plt.savefig('plt/rand_forest_elbow_graph_{}.png'.format(plotSuffix))
        # plt.show()
    return gs.best_params_


def plotBestParams(X_train, y_train):

    # param: n_estimators
    rf = RandomForestRegressor(random_state=0)
    param_grid = {
        'n_estimators': list(range(1, 101, 1)),
    }
    print('grid: \n', param_grid)
    best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'n_estimators')
    
    # # param: max_features, best params: {'max_features': 21}
    # rf = RandomForestRegressor(random_state=0)
    # param_grid = {
    #     'max_features': list(range(1, X_train.shape[1], 1))
    # }
    # print('grid: \n', param_grid)
    # best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'max_features')
    
    # # param: mix
    # rf = RandomForestRegressor(random_state=0)
    # param_grid = {
    #     'n_estimators': list(range(16, 25, 1)),
    #     'max_features': list(range(1, len(X_train.columns), 1))
    # }
    # print('grid: \n', param_grid)
    # best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, plotSuffix = 'mix')

    return best_params

def printPerformance(rf, X_test, y_test, X_columns):
    print('random forest accuracy:                   ', rf.score(X_test, y_test))
    # print('prediction:                               ', rf.predict(X_test[0].reshape(1, -1)))
    # print('true value:                               ', y_test[0])
    
    ft_imp = pd.Series(rf.feature_importances_, index=X_columns).sort_values(ascending=False)
    print()
    print('feature importance:                   \n', ft_imp.head(10))
    
    y_pred = rf.predict(X_test)
    y_true = y_test
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 1 - mape
    print()
    print('Mean Absolute Error (MAE):               ', metrics.mean_absolute_error(y_true, y_pred))
    print('Mean Squared Error (MSE):                ', metrics.mean_squared_error(y_true, y_pred))
    print('Root Mean Squared Error (RMSE):          ', metrics.mean_squared_error(y_true, y_pred, squared=False))
    print('Mean Absolute Percentage Error (MAPE):   ', mape)
    print('Accuracy (1 - MAPE):                     ', accuracy)
    print('Explained Variance Score:                ', metrics.explained_variance_score(y_true, y_pred))
    # print('Max Error:                               ', metrics.max_error(y_true, y_pred))
    # print('Mean Squared Log Error:                  ', metrics.mean_squared_log_error(y_true, y_pred))
    print('Median Absolute Error:                   ', metrics.median_absolute_error(y_true, y_pred))
    print('R^2:                                     ', metrics.r2_score(y_true, y_pred))
    # print('Mean Poisson Deviance:                   ', metrics.mean_poisson_deviance(y_true, y_pred))
    # print('Mean Gamma Deviance:                     ', metrics.mean_gamma_deviance(y_true, y_pred))


if __name__ == "__main__":
    train_mean = pd.read_csv('data/Universities_train_mean.csv')
    train_median = pd.read_csv('data/Universities_train_median.csv')

    test_mean = pd.read_csv('data/Universities_test_mean.csv')
    test_median = pd.read_csv('data/Universities_test_median.csv')

    # separate data frame to X and y
    y_columns = ['UG_average_fees_(in_pounds)', 'PG_average_fees_(in_pounds)']
    X_columns = train_mean.drop(columns=y_columns).columns
    X_train_mean = train_mean.drop(columns=y_columns).to_numpy()
    y_train_mean = train_mean[y_columns].to_numpy()

    X_test_mean = test_mean.drop(columns=y_columns).to_numpy()
    y_test_mean = test_mean[y_columns].to_numpy()
    results = []

    # plotBestParams(X_train_mean, y_train_mean)
    # best params: {'max_features': 10, 'n_estimators': 22}
    
    rf = RandomForestRegressor(n_estimators=22, max_features=10)
    rf.fit(X_train_mean, y_train_mean)

    printPerformance(rf, X_test_mean, y_test_mean, X_columns)
    y_pred = rf.predict(X_test_mean)


    # crate a data frame with the results of the selected model
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1],
                                    "truth UG_average_fees_(in_pounds)": y_test_mean[:, 0], "truth PG_average_fees_(in_pounds)": y_test_mean[:, 1]})
    
    print(predicted_truth)
    predicted_truth.to_csv("models/rf_predicted_truth.csv")