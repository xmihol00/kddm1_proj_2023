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
parser.add_argument("-ds", "--dataset_seed", type=int, default=RANDOM_SEED, help="Random seed of the data set splitting.")
args = parser.parse_args()

results = {
    "mean_all": [],
    "median_all": [],
    "mixed_all": [],
    "mean_continuous": [],
    "median_continuous": [],
    "mixed_continuous": [],
    "mean_selected": [],
    "median_selected": [],
    "mixed_selected": [],
}

continuous_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                      "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                      "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]
selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                    "Academic_staff_from", "Academic_staff_to"]

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

        # separate data frame to X and y, train with all columns
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
        results["mean_all"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on median imputed data
        model_median = LinearRegression()
        model_median.fit(X_train_median, y_train)
        # predict
        y_pred = model_median.predict(X_test_median)
        results["median_all"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on mixed imputed data
        model_mixed = LinearRegression()
        model_mixed.fit(X_train_mixed, y_train)
        # predict
        y_pred = model_mixed.predict(X_test_mixed)
        results["mixed_all"].append(metrics.mean_squared_error(y_test, y_pred))

        # train the models with all continuous columns
        X_train_mean = train_mean[continuous_columns].to_numpy()
        X_train_median = train_median[continuous_columns].to_numpy()
        X_train_mixed = train_mixed[continuous_columns].to_numpy()

        X_test_mean = test_mean[continuous_columns].to_numpy()
        X_test_median = test_median[continuous_columns].to_numpy()
        X_test_mixed = test_mixed[continuous_columns].to_numpy()

        # train on mean imputed data
        model_mean = LinearRegression()
        model_mean.fit(X_train_mean, y_train)
        # predict
        y_pred = model_mean.predict(X_test_mean)
        results["mean_continuous"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on median imputed data
        model_median = LinearRegression()
        model_median.fit(X_train_median, y_train)
        # predict
        y_pred = model_median.predict(X_test_median)
        results["median_continuous"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on mixed imputed data
        model_mixed = LinearRegression()
        model_mixed.fit(X_train_mixed, y_train)
        # predict
        y_pred = model_mixed.predict(X_test_mixed)
        results["mixed_continuous"].append(metrics.mean_squared_error(y_test, y_pred))

        # train the models with selected columns
        X_train_mean = train_mean[selected_columns].to_numpy()
        X_train_median = train_median[selected_columns].to_numpy()
        X_train_mixed = train_mixed[selected_columns].to_numpy()

        X_test_mean = test_mean[selected_columns].to_numpy()
        X_test_median = test_median[selected_columns].to_numpy()
        X_test_mixed = test_mixed[selected_columns].to_numpy()

        # train on mean imputed data
        model_mean = LinearRegression()
        model_mean.fit(X_train_mean, y_train)
        # predict
        y_pred = model_mean.predict(X_test_mean)
        results["mean_selected"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on median imputed data
        model_median = LinearRegression()
        model_median.fit(X_train_median, y_train)
        # predict
        y_pred = model_median.predict(X_test_median)
        results["median_selected"].append(metrics.mean_squared_error(y_test, y_pred))

        # train on mixed imputed data
        model_mixed = LinearRegression()
        model_mixed.fit(X_train_mixed, y_train)
        # predict
        y_pred = model_mixed.predict(X_test_mixed)
        results["mixed_selected"].append(metrics.mean_squared_error(y_test, y_pred))
    
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
    train = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mixed.csv")
    test = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mixed.csv")

    # separate data frame to X and y
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
    X_train = train[continuous_columns].to_numpy()
    X_test = test[continuous_columns].to_numpy()

    y_train = train[target_columns].to_numpy()
    y_test = test[target_columns].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # store predicted and ground truth values of the best imputed data
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1], 
                                    "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})

    print("Predicted vs. ground truth values mixed imputed on continuous columns:")
    print(predicted_truth)
    predicted_truth.to_csv(f"results/baseline_predicted_truth_seed{args.dataset_seed}.csv", index=False)

    # print results table
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    print("\nResults:")
    print("  - MSE:     ", mse)
    print("  - MAE:     ", mae)
    print("  - RMSE:    ", rmse)
    print("  - R2 Score:", r2)
