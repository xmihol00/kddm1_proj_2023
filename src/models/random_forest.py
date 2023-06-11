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
from constants import RANDOM_SEED, DATA_PATH, CROSS_VALIDATION_SEED, CV_SPLITS

PERFORM_CROSS_VALIDATION = RANDOM_SEED == CROSS_VALIDATION_SEED
IS_ANALYZING_PARAMETERS_BY_ELBOW_METHOD = False

def performGridSearch(X_train, y_train, param_grid: dict, create_Plot = False, datasetSuffix = ''):
    # use grid search to find best params with 3-fold cross validation
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=CV_SPLITS)
    gs.fit(X_train, y_train)
    print(f'{datasetSuffix} best params:', gs.best_params_)

    # Elbow Graph
    # optimizes accuracy without adding unnecessary performance complexity
    scores = gs.cv_results_['mean_test_score']
    max_score = np.round(scores.max(), 3)

    if create_Plot:
        plt.figure()
        if len(param_grid.values()) == 1:
            x = list(param_grid.values())[0]
            plt.plot(x, scores)
            plt.vlines(x[np.argmax(scores)], scores.min(), scores.max(), color='g')
        else:
            plt.plot(range(len(scores)), scores)
            plt.vlines(np.argmax(scores), scores.min(), scores.max(), color='g')
        plt.grid()
        plt.title(f"best param: {gs.best_params_}\nscore: {max_score}")
        plt.xlabel('params')
        plt.ylabel('neg_mean_squared_error')
        plt.savefig('plots/rand_forest_elbow_graph_{}.png'.format(datasetSuffix))
        # plt.show()
    return gs.best_params_

def evaluatePerformanceOnCV(X_train: np.ndarray, y_train: np.ndarray, best_params: dict, datasetSuffix: str):
    # 3-fold cross validation
    results_mse = []
    results_mae = []
    for train_index, test_index in ms.KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        
        # train model
        rf = RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=best_params.get('n_estimators', 100), max_features=best_params.get('max_features', 1.0))
        rf.fit(X_train_fold, y_train_fold)

        # evaluate model
        y_pred = rf.predict(X_val_fold)
        mse = metrics.mean_squared_error(y_val_fold, y_pred)
        mae = metrics.mean_absolute_error(y_val_fold, y_pred)
        results_mse.append(mse)
        results_mae.append(mae)
    
    # average results
    avg_mse = np.mean(results_mse)
    avg_mae = np.mean(results_mae)
    print(f'{datasetSuffix} average MSE: {avg_mse}, average MEA: {avg_mae}', )

def plotBestParams(X_train: np.ndarray, y_train: np.ndarray, suffix: str):
    print()
    # param: max_features, best params: {'max_features': 12}
    param_grid = {
        'max_features': list(range(1, X_train.shape[1], 1))
    }
    # print('grid: \n', param_grid)
    performGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_max_features')
    
    # param: n_estimators, best params: {'n_estimators': 88}
    param_grid = {
        'n_estimators': list(range(10, 101, 1)),
    }
    # print('grid: \n', param_grid)
    performGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_n_estimators')
    
    # param: mix, best params: {'max_features': 17, 'n_estimators': 18}
    max_features_upper_limit = np.min((X_train.shape[1], 18))
    param_grid = {
        'max_features': list(range(1, max_features_upper_limit)),
        'n_estimators': list(range(10, 101, 2)),
    }
    # print('grid: \n', param_grid)
    best_params = performGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_mix')
    return best_params

def printFeatureImportance(rf: RandomForestRegressor, columns):
    n = len(columns)
    ft_imp = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    feature_column_idx = [list(columns).index(feature) if feature in columns else None for feature in ft_imp[:n].index]
    print()
    print("Feature importance:")
    print(pd.DataFrame({
        'pos':range(1, n+1),
        'ft_imp':ft_imp.head(n).round(3),
        'column_index': feature_column_idx
        }))
    
    print('column index of features by relevance:   ', feature_column_idx[:n])
    print('column index of features by index order: ', sorted(feature_column_idx[:n]))
    
    # columns_origin = pd.read_csv(DATA_PATH["original"] + 'Universities.csv').columns
    # feature_column_idx = []
    # for feature in ft_imp[:n].index:
    #     feature_first_word = str(feature).split('_')[0]
    #     for i, col in enumerate(columns_origin):
    #         if str(col).startswith(feature_first_word):
    #             feature_column_idx.append(i)
    #             continue
    
    # _, idx = np.unique(feature_column_idx, return_index=True)
    
    # print()
    # print('column index of   origin by relevance:          ', feature_column_idx)
    # print('column index of   origin by relevance unique:   ', list(np.array(feature_column_idx)[np.sort(idx)]))
    # print('column index of   origin by index order:        ', sorted(feature_column_idx))
    # print('column index of   origin by index order unique: ', sorted(set(feature_column_idx)))

def printPerformance(rf: RandomForestRegressor, X_test, y_test):
    printFeatureImportance(rf, selected_columns)
    
    y_pred = rf.predict(X_test)
    y_true = y_test
    print()
    print('Performance of the best model on the test set:')
    print('  Accuracy:                                {:16.3f}'.format(rf.score(X_test, y_test)))
    print()
    print('  Mean Squared Error (MSE):                {:16.2f}'.format(metrics.mean_squared_error(y_true, y_pred)))
    print('  Root Mean Squared Error (RMSE):          {:16.2f}'.format(metrics.mean_squared_error(y_true, y_pred, squared=False)))
    print('  Mean Absolute Error (MAE):               {:16.2f}'.format(metrics.mean_absolute_error(y_true, y_pred)))
    print('  R^2:                                     {:16.3f}'.format(metrics.r2_score(y_true, y_pred)))

    return y_pred

def evaluateMean(data: "dict[pd.DataFrame]", X_columns: "list[str]", y_columns: "list[str]", param_grid: dict, suffix: str):
        X = data["train_mean"][X_columns].to_numpy()
        y = data["train_mean"][y_columns].to_numpy()

        best_parameters = performGridSearch(X, y, param_grid, create_Plot=True, datasetSuffix=f'mean_{suffix}')
        evaluatePerformanceOnCV(X, y, best_parameters, datasetSuffix=f'mean {suffix} columns')

def evaluateMedian(data: "dict[pd.DataFrame]", X_columns: "list[str]", y_columns: "list[str]", param_grid: dict, suffix: str):
        X = data["train_median"][X_columns].to_numpy()
        y = data["train_median"][y_columns].to_numpy()

        best_parameters = performGridSearch(X, y, param_grid, create_Plot=True, datasetSuffix=f'median_{suffix}')
        evaluatePerformanceOnCV(X, y, best_parameters, datasetSuffix=f'median {suffix} columns')

def evaluateMix(data: "dict[pd.DataFrame]", X_columns: "list[str]", y_columns: "list[str]", param_grid: dict, suffix: str):
        X = data["train_mixed"][X_columns].to_numpy()
        y = data["train_mixed"][y_columns].to_numpy()

        best_parameters = performGridSearch(X, y, param_grid, create_Plot=True, datasetSuffix=f'mixed_{suffix}')
        evaluatePerformanceOnCV(X, y, best_parameters, datasetSuffix=f'mixed {suffix} columns')

    
if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    
    np.random.seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)

    data = {
        "train_mean"    : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mean.csv'),
        "train_median"  : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_median.csv'),
        "train_mixed"   : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mixed.csv'),
        "test_mean"     : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mean.csv'),
        "test_median"   : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_median.csv'),
        "test_mixed"    : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mixed.csv'),
    }
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
    print("shape: {}".format(data["train_mixed"].shape))
    print("seed: {RANDOM_SEED}\n")

    if PERFORM_CROSS_VALIDATION:
        # grid search and cross validation evaluation on selected columns
        X_columns = list(set(data["train_mean"].columns) - set(target_columns))
        param_grid = {
            'max_features': list(range(1, 18, 1)),
            'n_estimators': list(range(80, 101, 2)),
        }
        evaluateMean(data, X_columns, target_columns, param_grid, "all")
        evaluateMedian(data, X_columns, target_columns, param_grid, "all")
        evaluateMix(data, X_columns, target_columns, param_grid, "all")

        X_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                            "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                            "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]
        param_grid = {
            'max_features': list(range(1, len(X_columns), 1)),
            'n_estimators': list(range(80, 101, 2)),
        }
        evaluateMean(data, X_columns, target_columns, param_grid, "continuous")
        evaluateMedian(data, X_columns, target_columns, param_grid, "continuous")
        evaluateMix(data, X_columns, target_columns, param_grid, "continuous")

        X_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                            "Academic_staff_from", "Academic_staff_to"]
        param_grid = {
            'max_features': list(range(1, len(X_columns), 1)),
            'n_estimators': list(range(80, 101, 2)),
        }
        evaluateMean(data, X_columns, target_columns, param_grid, "selected")
        evaluateMedian(data, X_columns, target_columns, param_grid, "selected")
        evaluateMix(data, X_columns, target_columns, param_grid, "selected")

    selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                            "Academic_staff_from", "Academic_staff_to"]

    X_train = data["train_mixed"][selected_columns].to_numpy()
    y_train = data["train_mixed"][target_columns].to_numpy()
    X_test  = data["test_mixed"][selected_columns].to_numpy()
    y_test  = data["test_mixed"][target_columns].to_numpy()

    if IS_ANALYZING_PARAMETERS_BY_ELBOW_METHOD:
        # analyses singe parameter performance by plots
        # best performance on centered max_features and n_estimators above 80
        plotBestParams(X_train, y_train, suffix="mixed_all")

    # fit RF with best parameters found during cross validation
    max_features=1
    n_estimators=98
    rf = RandomForestRegressor(random_state=RANDOM_SEED, max_features=max_features, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    print()
    print(f"RF model on mixed dataset with max_features: {max_features}, n_estimators: {n_estimators}")
    
    # evaluate performance on test set
    y_pred = printPerformance(rf, X_test, y_test)

    # crate a data frame with the results of the selected model
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1],
                                    "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})
    
    print(predicted_truth)

    # save the results
    if PERFORM_CROSS_VALIDATION:
        predicted_truth.to_csv("results/RF_predicted_truth_CV.csv", index=False)
    else:
        predicted_truth.to_csv(f"results/RF_predicted_truth_seed{RANDOM_SEED}.csv", index=False)
