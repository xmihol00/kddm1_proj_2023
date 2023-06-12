import argparse
import json
import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH, RANDOM_SEED, DATASET_RANDOM_SEEDS

parser = argparse.ArgumentParser()
parser.add_argument("-cv", "--cross_validation", action="store_true", help="Perform cross-validation.")
args = parser.parse_args()

results = {
    "mean": [],
    "median": [],
    "mixed": [],
}

if args.cross_validation:
    for random_seed in DATASET_RANDOM_SEEDS:
        # preprocess data set for current random seed
        os.system(f"RANDOM_SEED={random_seed} ./run_preprocessing.sh")
        
        # load data set
        train_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mean.csv")
        train_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
        train_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mixed.csv")

        test_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mean.csv")
        test_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")
        test_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mixed.csv")

        # separate data frame to X and y
        target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
        X_train_mean = train_mean.drop(columns=target_columns).to_numpy()
        X_train_median = train_median.drop(columns=target_columns).to_numpy()
        X_train_mixed = train_mixed.drop(columns=target_columns).to_numpy()

        X_test_mean = test_mean.drop(columns=target_columns).to_numpy()
        X_test_median = test_median.drop(columns=target_columns).to_numpy()
        X_test_mixed = test_mixed.drop(columns=target_columns).to_numpy()

        y_train = train_mean[target_columns].to_numpy()
        y_test = test_mean[target_columns].to_numpy()

        # train on mean imputed data
        model_mean = LinearRegression()
        model_mean.fit(X_train_mean, y_train)

        # predict
        y_pred = model_mean.predict(X_test_mean)

        # evaluate
        results["mean"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on median imputed data
        model_median = LinearRegression()
        model_median.fit(X_train_median, y_train)

        # predict
        y_pred = model_median.predict(X_test_median)

        # evaluate
        results["median"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on mixed imputed data
        model_mixed = LinearRegression()
        model_mixed.fit(X_train_mixed, y_train)

        # predict
        y_pred = model_mixed.predict(X_test_mixed)

        # evaluate
        results["mixed"].append(metrics.mean_squared_error(y_test, y_pred))
    
    print("CV results MSE:", json.dumps(results, indent=2), sep="\n", end="\n\n")
    # average results
    best_mse = sys.maxsize
    best_key = ""
    for key in results.keys():
        results[key] = np.median(results[key])
        if results[key] < best_mse:
            best_mse = results[key]
            best_key = key
    
    print("CV average results MSE:", json.dumps(results, indent=2), sep="\n", end="\n\n")
    print("Best model:", f"{best_key} - MSE: {results[best_key]}", sep="\n")

else:
    # load data set
    train = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
    test = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")

    # separate data frame to X and y
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
    X_train = train.drop(columns=target_columns).to_numpy()
    X_test = test.drop(columns=target_columns).to_numpy()


    y_train = train[target_columns].to_numpy()
    y_test = test[target_columns].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # store predicted and ground truth values of the best imputed data
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1], 
                                    "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})

    print("Predicted vs. ground truth values mixed imputed:")
    print(predicted_truth)
    predicted_truth.to_csv(f"results/baseline_predicted_truth_seed{RANDOM_SEED}.csv", index=False)

    # print results table
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    print("\nResults:")
    print("  - MSE:     ", mse)
    print("  - RMSE:    ", rmse)
    print("  - MAE:     ", mae)
    print("  - R2 Score:", r2)
