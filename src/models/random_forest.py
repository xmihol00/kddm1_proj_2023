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
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH, CV_SPLITS, RANDOM_SEED, DATASET_RANDOM_SEEDS

def plotGridSearch(param_grid: dict, params: "list[dict]", scores: list, datasetSuffix: str):
    max_score = np.round(scores.max(), 3)
    x = [str(list(dict.values())) for dict in params]
    step_size = 1
    for k, v in reversed(list(param_grid.items())):
        if step_size >= len(params) // 20:
            break
        step_size *= len(v)

    best_score_idx = np.argmax(scores)
    best_params = params[best_score_idx]

    fig, ax = plt.subplots()
    ax.plot(x, scores)
    ax.vlines(best_score_idx, scores.min(), scores.max(), color='g')
    ax
    ax.xaxis.set_ticks(x[0::step_size])
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.grid()
    plt.title(f"best param: {best_params}\nscore: {max_score}", fontsize=10)
    plt.xlabel('params')
    plt.ylabel('neg_mean_squared_error')
    plt.tight_layout()
    plt.savefig('plots/rand_forest_elbow_graph_{}.png'.format(datasetSuffix))
    # plt.show()

def getParamAndPlotGridSearch(X_train, y_train, param_grid: dict, create_Plot = False, datasetSuffix = ''):
    # use grid search to find best params with 3-fold cross validation
    setSeed()
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=CV_SPLITS)
    gs.fit(X_train, y_train)
    print(f'{datasetSuffix} best params:', gs.best_params_)

    # Elbow Graph
    # optimizes accuracy without adding unnecessary performance complexity
    params = gs.cv_results_['params']
    scores = gs.cv_results_['mean_test_score']
    if create_Plot:
        plotGridSearch(param_grid, params, scores, datasetSuffix)

    return gs.best_params_

def evaluatePerformanceOnCV(X_train: np.ndarray, y_train: np.ndarray, best_params: dict, datasetSuffix: str, seed: int = RANDOM_SEED, isPlotEnabled: bool = True):
    # 3-fold cross validation
    results_mse = []
    results_mae = []
    for train_index, test_index in ms.KFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        
        # train model
        setSeed()
        rf = RandomForestRegressor(random_state=seed, n_estimators=best_params.get('n_estimators', 100), max_features=best_params.get('max_features', 1.0))
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
    if isPlotEnabled:
        print(f'{datasetSuffix} average MSE: {avg_mse}, average MEA: {avg_mae}', )
    return avg_mse, avg_mae

def plotBestParams(X_train: np.ndarray, y_train: np.ndarray, suffix: str):
    print()
    # param: max_features, best params: {'max_features': 12}
    param_grid = {
        'max_features': list(range(1, X_train.shape[1], 1)),
        'random_state': [RANDOM_SEED],
    }
    # print('grid: \n', param_grid)
    getParamAndPlotGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_max_features')
    
    # param: n_estimators, best params: {'n_estimators': 88}
    param_grid = {
        'n_estimators': list(range(10, 101, 1)),
        'random_state': [RANDOM_SEED],
    }
    # print('grid: \n', param_grid)
    getParamAndPlotGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_n_estimators')
    
    # param: mix, best params: {'max_features': 17, 'n_estimators': 18}
    max_features_upper_limit = np.min((X_train.shape[1], 18))
    param_grid = {
        'max_features': list(range(1, max_features_upper_limit)),
        'n_estimators': list(range(10, 101, 2)),
        'random_state': [RANDOM_SEED],
    }
    # print('grid: \n', param_grid)
    best_params = getParamAndPlotGridSearch(X_train, y_train, param_grid, create_Plot = True, datasetSuffix = f'{suffix}_param_mix')
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

def printPerformance(rf: RandomForestRegressor, X_test, y_test, x_col):
    printFeatureImportance(rf, x_col)
    
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

def evaluate(X: np.ndarray, y: np.ndarray, param_grid: dict, suffix: str):
        X = X.to_numpy()
        y = y.to_numpy()
        
        best_parameters = getParamAndPlotGridSearch(X, y, param_grid, create_Plot=True, datasetSuffix=suffix)
        evaluatePerformanceOnCV(X, y, best_parameters, datasetSuffix=f'{suffix}_columns')

def evaluateMultipleSeeds(X: np.ndarray, y: np.ndarray, param_grid: dict, suffix: str):
        X = X.to_numpy()
        y = y.to_numpy()
        
        param_grid_size = np.product([len(v) for v in param_grid.values()])
        seed_range = len(DATASET_RANDOM_SEEDS)
        scores_arr = np.zeros((seed_range, param_grid_size))
        for i, seed in enumerate(DATASET_RANDOM_SEEDS):
            param_grid["random_state"] = [seed]
            
            setSeed()
            rf = RandomForestRegressor(random_state=seed)
            gs = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=CV_SPLITS)
            gs.fit(X, y)

            scores_arr[i, :] = gs.cv_results_['mean_test_score']
            
        scores_median = np.median(scores_arr, axis=0)
        best_score_idx = np.argmax(scores_median)
        params = gs.cv_results_['params']
        for d in params:
            del d["random_state"]
        
        best_parameters = params[best_score_idx]
        print(f'{suffix} best params:', best_parameters)
        plotGridSearch(param_grid, params=params, scores=scores_median, datasetSuffix=suffix)
        
        ave_mse_list = [0] * seed_range
        ave_mae_list = [0] * seed_range
        for i, seed in enumerate(DATASET_RANDOM_SEEDS):
            ave_mse_list[i], ave_mae_list[i] = evaluatePerformanceOnCV(X, y, best_parameters, datasetSuffix='', seed=seed, isPlotEnabled=False)

        median_ave_mse = np.median(ave_mse_list)
        median_ave_mae = np.median(ave_mae_list)
        print(f'{suffix} average MSE: {median_ave_mse}, average MEA: {median_ave_mae}', )

def rf_param_search_single_seed(data: "dict[pd.DataFrame]", columns: "dict[dict[list]]", param_grid: "dict[dict[list]]"):
    # grid search and cross validation evaluation on selected columns
    evaluate(  data["train_mean"][columns["all"]],   data["train_mean"][columns["target"]], param_grid["all"], "all")
    evaluate(data["train_median"][columns["all"]], data["train_median"][columns["target"]], param_grid["all"], "all")
    evaluate( data["train_mixed"][columns["all"]],  data["train_mixed"][columns["target"]], param_grid["all"], "all")
    
    evaluate(  data["train_mean"][columns["continuous"]],   data["train_mean"][columns["target"]], param_grid["continuous"], "continuous")
    evaluate(data["train_median"][columns["continuous"]], data["train_median"][columns["target"]], param_grid["continuous"], "continuous")
    evaluate( data["train_mixed"][columns["continuous"]],  data["train_mixed"][columns["target"]], param_grid["continuous"], "continuous")
    
    evaluate(  data["train_mean"][columns["selected"]],   data["train_mean"][columns["target"]], param_grid["selected"], "selected")
    evaluate(data["train_median"][columns["selected"]], data["train_median"][columns["target"]], param_grid["selected"], "selected")
    evaluate( data["train_mixed"][columns["selected"]],  data["train_mixed"][columns["target"]], param_grid["selected"], "selected")
    
def rf_param_search_multiple_seeds(data: "dict[pd.DataFrame]", columns: "dict[dict[list]]", param_grid: "dict[dict[list]]"):
    # grid search and cross validation evaluation on selected columns
    # evaluateMultipleSeeds(  data["train_mean"][columns["all"]],   data["train_mean"][columns["target"]], param_grid["all"], "all_multiple_seeds")
    # evaluateMultipleSeeds(data["train_median"][columns["all"]], data["train_median"][columns["target"]], param_grid["all"], "all_multiple_seeds")
    # evaluateMultipleSeeds( data["train_mixed"][columns["all"]],  data["train_mixed"][columns["target"]], param_grid["all"], "all_multiple_seeds")
    
    # evaluateMultipleSeeds(  data["train_mean"][columns["continuous"]],   data["train_mean"][columns["target"]], param_grid["continuous"], "continuous_multiple_seeds")
    # evaluateMultipleSeeds(data["train_median"][columns["continuous"]], data["train_median"][columns["target"]], param_grid["continuous"], "continuous_multiple_seeds")
    # evaluateMultipleSeeds( data["train_mixed"][columns["continuous"]],  data["train_mixed"][columns["target"]], param_grid["continuous"], "continuous_multiple_seeds")
    
    # evaluateMultipleSeeds(  data["train_mean"][columns["selected"]],   data["train_mean"][columns["target"]], param_grid["selected"], "selected_multiple_seeds")
    # evaluateMultipleSeeds(data["train_median"][columns["selected"]], data["train_median"][columns["target"]], param_grid["selected"], "selected_multiple_seeds")
    evaluateMultipleSeeds( data["train_mixed"][columns["selected"]],  data["train_mixed"][columns["target"]], param_grid["selected"], "selected_multiple_seeds")
    
def rf_with_best_param(data, columns):
    best_columns = columns["all"]
    X_train = data["train_mixed"][best_columns].to_numpy()
    y_train = data["train_mixed"][columns["target"]].to_numpy()
    X_test  = data["test_mixed"][best_columns].to_numpy()
    y_test  = data["test_mixed"][columns["target"]].to_numpy()

    if IS_ANALYZING_PARAMETERS_BY_ELBOW_METHOD:
        # analyses singe parameter performance by plots
        # best performance on centered max_features and n_estimators above 80
        plotBestParams(X_train, y_train, suffix="mixed_all")

    # fit RF with best parameters found during cross validation
    max_features=6
    n_estimators=100
    setSeed()
    rf = RandomForestRegressor(random_state=RANDOM_SEED, max_features=max_features, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    print()
    print(f"RF model on mixed dataset with random_state: {RANDOM_SEED}, max_features: {max_features}, n_estimators: {n_estimators}")
    
    # evaluate performance on test set
    y_pred = printPerformance(rf, X_test, y_test, best_columns)

    # crate a data frame with the results of the selected model
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1],
                                    "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})
    
    print(predicted_truth)

    return predicted_truth

def getModelData() -> "list[dict[pd.DataFrame], dict[dict[list]], dict[dict[list]]]":
    data = {
        "train_mean"    : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mean.csv'),
        "train_median"  : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_median.csv'),
        "train_mixed"   : pd.read_csv(DATA_PATH["numeric"] + 'Universities_train_mixed.csv'),
        "test_mean"     : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mean.csv'),
        "test_median"   : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_median.csv'),
        "test_mixed"    : pd.read_csv(DATA_PATH["numeric"] + 'Universities_test_mixed.csv'),
    }
    columns = {
        "target":       ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"],
        "all":          list(set(data["train_mean"].columns) - set(("UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"))),
        "continues":    ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                         "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                         "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"],
        "selected":     ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                         "Academic_staff_from", "Academic_staff_to"],
    }
    param_grid = {
        "all": {
            'max_features': list(range(1, 18, 1)),
            'n_estimators': list(range(80, 101, 2)),
            'random_state': [RANDOM_SEED],
        },
        "continues": {
            'max_features': list(range(1, len(columns["continues"]), 1)),
            'n_estimators': list(range(80, 101, 2)),
            'random_state': [RANDOM_SEED],
        },
        "selected": {
            'max_features': list(range(1, len(columns["selected"]), 1)),
            'n_estimators': list(range(80, 101, 200)),
            'random_state': [RANDOM_SEED],
        },
    }
    return data, columns, param_grid

def setSeed():
    np.random.seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cv", "--cross_validation", action="store_true", help="Perform cross-validation.")
    args = parser.parse_args()
    # args.cross_validation = True
    
    os.makedirs("plots", exist_ok=True)
    setSeed()

    IS_ANALYZING_PARAMETERS_BY_ELBOW_METHOD = False
    IS_CALCULATE_ON_SINGLE_SEED = False

    data, columns, param_grid = getModelData()

    print("shape: {}".format(data["train_mixed"].shape))
    print(f"seed: {RANDOM_SEED}\n")
    
    # find and plot best params
    if args.cross_validation:
        if IS_CALCULATE_ON_SINGLE_SEED:
            rf_param_search_single_seed(data, columns, param_grid)
        else:
            rf_param_search_multiple_seeds(data, columns, param_grid)
    
    # result
    predicted_truth = rf_with_best_param(data, columns)
    
    # save the results
    if args.cross_validation:
        predicted_truth.to_csv("results/RF_predicted_truth_CV.csv", index=False)
    else:
        predicted_truth.to_csv(f"results/RF_predicted_truth_seed{RANDOM_SEED}.csv", index=False)
