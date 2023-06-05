#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import os, sys
import random as rn
from sklearn import model_selection as ms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import RANDOM_SEED, DATA_PATH

def performGridSearch(X_train, y_train, param_grid, create_Plot = False, datasetSuffix = ''):
    # use grid search to find best params with 3-fold cross validation
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=3)
    gs.fit(X_train, y_train)
    print(f'{datasetSuffix} best params:', gs.best_params_)

    # Elbow Graph
    # optimizes accuracy without adding unnecessary performance complexity
    scores = gs.cv_results_['mean_test_score']

    if create_Plot:
        plt.figure()
        plt.plot(range(len(scores)), scores)
        plt.grid()
        plt.xlabel('params')
        plt.ylabel('neg_mean_squared_error')
        plt.savefig('plots/rand_forest_elbow_graph_{}.png'.format(datasetSuffix))
        # plt.show()
    return gs.best_params_

def evaluatePerformanceOnCV(X_train: np.ndarray, y_train: np.ndarray, best_params: dict, datasetSuffix: str):
    #TODO: perform cross validation on 3 folds and average the results, print (use printPerformance) the results to stdout and 
    # then store them to logs/, visually inspect the results and rerun final training with best params and best dataset

    # 3-fold cross validation
    results = []
    for train_index, test_index in ms.KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        
        # train model
        rf = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=best_params['n_estimators'], max_features=best_params['max_features'])
        rf.fit(X_train_fold, y_train_fold)

        # evaluate model
        y_pred = rf.predict(X_val_fold)
        mse = metrics.mean_squared_error(y_val_fold, y_pred)
        results.append(mse)
    
    # average results
    avg_mse = np.mean(results)
    print(f'{datasetSuffix} avg mse: ', avg_mse)

def plotBestParams(X_train: np.ndarray, y_train: np.ndarray):
    # param: max_features, best params: {'max_features': 21}
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    param_grid = {
        'max_features': list(range(1, X_train.shape[1] + 2, 1))
    }
    print('grid: \n', param_grid)
    performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, datasetSuffix = 'max_features')
    
    # param: n_estimators
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    param_grid = {
        'n_estimators': list(range(1, 101, 1)),
    }
    print('grid: \n', param_grid)
    performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, datasetSuffix = 'n_estimators')
    
    # param: n_estimators
    # # best params: {'max_features': 17, 'n_estimators': 18}
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    param_grid = {
        'max_features': (10, 17),
        'n_estimators': list(range(1, 101, 1)),
    }
    print('grid: \n', param_grid)
    best_params = performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, datasetSuffix = 'mix1')
    
    # param: mix
    # best params: {'max_features': 17, 'n_estimators': 38}
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    param_grid = {
        'max_features': list(range(10, 17, 1)),
        'n_estimators': list(range(35, 60, 1)),
    }
    print('grid: \n', param_grid)
    performGridSearch(X_train, y_train, rf, param_grid, create_Plot = True, datasetSuffix = 'mix2')

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
    
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    
    np.random.seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)

    train_mean = pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mean.csv')
    train_median = pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_median.csv')
    train_mixed = pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mixed.csv')

    test_mean = pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mean.csv')
    test_median = pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_median.csv')
    test_mixed = pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mixed.csv')

    # separate data frame to X and y
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
    X_train_mean = train_mean.drop(columns=target_columns).to_numpy()
    y_train_mean = train_mean[target_columns].to_numpy()
    X_train_median = train_median.drop(columns=target_columns).to_numpy()
    y_train_median = train_median[target_columns].to_numpy()
    X_train_mixed = train_mixed.drop(columns=target_columns).to_numpy()
    y_train_mixed = train_mixed[target_columns].to_numpy()

    X_test_mean = test_mean.drop(columns=target_columns).to_numpy()
    y_test_mean = test_mean[target_columns].to_numpy()
    X_test_median = test_median.drop(columns=target_columns).to_numpy()
    y_test_median = test_median[target_columns].to_numpy()
    X_test_mixed = test_mixed.drop(columns=target_columns).to_numpy()
    y_test_mixed = test_mixed[target_columns].to_numpy()

    # TODO: specify meaningful parameter grid
    param_grid = {
        'max_features': list(range(1, 101, 10)),
        'n_estimators': list(range(1, 101, 10)),
    }

    # grid search and cross validation evaluation on all columns
    best_parameters = performGridSearch(X_train_mean, y_train_mean, param_grid, create_Plot=True, datasetSuffix='mean')
    evaluatePerformanceOnCV(X_train_mean, y_train_mean, best_parameters, datasetSuffix='mean')
    best_parameters = performGridSearch(X_train_median, y_train_median, param_grid, create_Plot=True, datasetSuffix='median')
    evaluatePerformanceOnCV(X_train_median, y_train_median, best_parameters, datasetSuffix='median')
    best_parameters = performGridSearch(X_train_mixed, y_train_mixed, param_grid, create_Plot=True, datasetSuffix='mixed')
    evaluatePerformanceOnCV(X_train_mixed, y_train_mixed, best_parameters, datasetSuffix='mixed')

    # grid search and cross validation evaluation on selected columns
    selected_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                        "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                        "International_students", "Academic_staff_from", "Academic_staff_to"]
    X_train_mean = train_mean[selected_columns].to_numpy()
    y_train_mean = train_mean[target_columns].to_numpy()
    X_train_median = train_median[selected_columns].to_numpy()
    y_train_median = train_median[target_columns].to_numpy()
    X_train_mixed = train_mixed[selected_columns].to_numpy()
    y_train_mixed = train_mixed[target_columns].to_numpy()

    X_test_mean = test_mean[selected_columns].to_numpy()
    y_test_mean = test_mean[target_columns].to_numpy()
    X_test_median = test_median[selected_columns].to_numpy()
    y_test_median = test_median[target_columns].to_numpy()
    X_test_mixed = test_mixed[selected_columns].to_numpy()
    y_test_mixed = test_mixed[target_columns].to_numpy()

    best_parameters = performGridSearch(X_train_mean, y_train_mean, param_grid, create_Plot=True, datasetSuffix='mean_selected')
    evaluatePerformanceOnCV(X_train_mean, y_train_mean, best_parameters, datasetSuffix='mean_selected')
    best_parameters = performGridSearch(X_train_median, y_train_median, param_grid, create_Plot=True, datasetSuffix='median_selected')
    evaluatePerformanceOnCV(X_train_median, y_train_median, best_parameters, datasetSuffix='median_selected')
    best_parameters = performGridSearch(X_train_mixed, y_train_mixed, param_grid, create_Plot=True, datasetSuffix='mixed_selected')
    evaluatePerformanceOnCV(X_train_mixed, y_train_mixed, best_parameters, datasetSuffix='mixed_selected')

    # grid search and cross validation evaluation on selected columns
    selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                        "Academic_staff_from", "Academic_staff_to"]

    X_train_mean = train_mean[selected_columns].to_numpy()
    y_train_mean = train_mean[target_columns].to_numpy()
    X_train_median = train_median[selected_columns].to_numpy()
    y_train_median = train_median[target_columns].to_numpy()
    X_train_mixed = train_mixed[selected_columns].to_numpy()
    y_train_mixed = train_mixed[target_columns].to_numpy()

    X_test_mean = test_mean[selected_columns].to_numpy()
    y_test_mean = test_mean[target_columns].to_numpy()
    X_test_median = test_median[selected_columns].to_numpy()
    y_test_median = test_median[target_columns].to_numpy()
    X_test_mixed = test_mixed[selected_columns].to_numpy()
    y_test_mixed = test_mixed[target_columns].to_numpy()

    best_parameters = performGridSearch(X_train_mean, y_train_mean, param_grid, create_Plot=True, datasetSuffix='mean_corr')
    evaluatePerformanceOnCV(X_train_mean, y_train_mean, best_parameters, datasetSuffix='mean_corr')
    best_parameters = performGridSearch(X_train_median, y_train_median, param_grid, create_Plot=True, datasetSuffix='median_corr')
    evaluatePerformanceOnCV(X_train_median, y_train_median, best_parameters, datasetSuffix='median_corr')
    best_parameters = performGridSearch(X_train_mixed, y_train_mixed, param_grid, create_Plot=True, datasetSuffix='mixed_corr')
    evaluatePerformanceOnCV(X_train_mixed, y_train_mixed, best_parameters, datasetSuffix='mixed_corr')

    # TODO: find the best parameters and best dataset and perform the final training and evaluation


    ## best params: {'max_features': 8}
    ## best params: {'n_estimators': 29}
    ## best params: {'max_features': 17, 'n_estimators': 37}
    #plotBestParams(X_train, y_train)
    #
    #rf = RandomForestRegressor(max_features=10, n_estimators=80)
    #rf.fit(X_train, y_train)
#
    #y_pred = rf.predict(X_test)
    #printPerformance(rf, X_test, y_test, columns)
#
    ## crate a data frame with the results of the selected model
    #predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1],
    #                                "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})
    #
    ## print(predicted_truth)
    #predicted_truth.to_csv("results/RF_predicted_truth.csv", index=False)
