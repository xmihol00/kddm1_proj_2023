import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras as nn
from sklearn import model_selection as ms
from keras import callbacks as tfcb
import random as rn
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH, CV_SPLITS, RANDOM_SEED, DATASET_RANDOM_SEEDS, MODEL_RANDOM_SEED

parser = argparse.ArgumentParser()
parser.add_argument("-cv", "--cross_validation", action="store_true", help="Perform cross-validation.")
args = parser.parse_args()

TRAIN_ALL_COLUMNS = True
TRAIN_ALL_CONTINUOS_COLUMNS = True
TRAIN_SELECTED_COLUMNS = True

def run_cross_validation(X_train, y_train, model, random_seed, verbose=0):
    cv_results = []
    cv_epochs = []
    patience = 25
    weights = model.get_weights()
    
    for train_index, test_index in ms.KFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_seed).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        
        # train the model on each fold
        model.set_weights(weights)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mse')
        history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=10000, batch_size=8, verbose=verbose, 
                        callbacks=[tfcb.EarlyStopping(monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)]).history
        # store the results
        cv_results.append(min(history['val_loss']))
        cv_epochs.append(len(history['val_loss']) - patience)
    
    # average the results
    epochs = int(np.average(cv_epochs))
    result = np.average(cv_results)
    if verbose:
        print(f"Cross validation results: {cv_results}")
        print(f"Cross validation average result: {result}")
        print(f"Average number of epochs: {epochs}")
    
    # return the model to the original weights
    model.set_weights(weights)
    return result, epochs

def train_and_evaluate(X_train, y_train, X_test, y_test, model, epochs, verbose=0):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=verbose)

    # evaluate the model on the test set
    pred_result = model.predict(X_test, verbose=verbose)
    eval_result = model.evaluate(X_test, y_test, verbose=verbose)
    if verbose:
        print(f"Results of a model with all columns.", f"  - loss: {eval_result}", 
              f"  - test samples (predicted, ground truth):", np.stack((pred_result, y_test)).T[:10], sep="\n")
    
    return pred_result, eval_result

if __name__ == "__main__":
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]

    continuous_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                          "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                          "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]
    selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students", 
                        "Academic_staff_from", "Academic_staff_to"]

    results_epochs = {
        "mean_2l_all": [],
        "mean_3l_all": [],
        "mean_4l_all": [],
        "median_2l_all": [],
        "median_3l_all": [],
        "median_4l_all": [],
        "mixed_2l_all": [],
        "mixed_3l_all": [],
        "mixed_4l_all": [],

        "mean_2l_continuous": [],
        "mean_3l_continuous": [],
        "mean_4l_continuous": [],
        "median_2l_continuous": [],
        "median_3l_continuous": [],
        "median_4l_continuous": [],
        "mixed_2l_continuous": [],
        "mixed_3l_continuous": [],
        "mixed_4l_continuous": [],

        "mean_2l_selected": [],
        "mean_3l_selected": [],
        "mean_4l_selected": [],
        "median_2l_selected": [],
        "median_3l_selected": [],
        "median_4l_selected": [],
        "mixed_2l_selected": [],
        "mixed_3l_selected": [],
        "mixed_4l_selected": []
    }
    results_mse = {
        "mean_2l_all": [],
        "mean_3l_all": [],
        "mean_4l_all": [],
        "median_2l_all": [],
        "median_3l_all": [],
        "median_4l_all": [],
        "mixed_2l_all": [],
        "mixed_3l_all": [],
        "mixed_4l_all": [],

        "mean_2l_continuous": [],
        "mean_3l_continuous": [],
        "mean_4l_continuous": [],
        "median_2l_continuous": [],
        "median_3l_continuous": [],
        "median_4l_continuous": [],
        "mixed_2l_continuous": [],
        "mixed_3l_continuous": [],
        "mixed_4l_continuous": [],

        "mean_2l_selected": [],
        "mean_3l_selected": [],
        "mean_4l_selected": [],
        "median_2l_selected": [],
        "median_3l_selected": [],
        "median_4l_selected": [],
        "mixed_2l_selected": [],
        "mixed_3l_selected": [],
        "mixed_4l_selected": []
    }

    if args.cross_validation:
        for random_seed in DATASET_RANDOM_SEEDS:
            # preprocess data set for current random seed
            os.system(f"RANDOM_SEED={random_seed} ./run_preprocessing.sh")

            # load the data set
            train_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mean.csv")
            train_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
            train_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mixed.csv")

            test_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mean.csv")
            test_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")
            test_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mixed.csv")

            # separate y (same for all data sets) from data frames
            y_train = train_mean[target_columns].to_numpy()
            y_test = test_mean[target_columns].to_numpy()

        ########################################### All columns ###########################################
            if TRAIN_ALL_COLUMNS:
                # train the models with all columns
                X_train_mean = train_mean.drop(columns=target_columns).to_numpy()
                X_train_median = train_median.drop(columns=target_columns).to_numpy()
                X_train_mixed = train_mixed.drop(columns=target_columns).to_numpy()

                X_test_mean = test_mean.drop(columns=target_columns).to_numpy()
                X_test_median = test_median.drop(columns=target_columns).to_numpy()
                X_test_mixed = test_mixed.drop(columns=target_columns).to_numpy()
                
                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_2l_all"].append(epochs)
                results_mse["mean_2l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_2l_all"].append(epochs)
                results_mse["median_2l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_2l_all"].append(epochs)
                results_mse["mixed_2l_all"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_3l_all"].append(epochs)
                results_mse["mean_3l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_3l_all"].append(epochs)
                results_mse["median_3l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_3l_all"].append(epochs)
                results_mse["mixed_3l_all"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_4l_all"].append(epochs)
                results_mse["mean_4l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_4l_all"].append(epochs)
                results_mse["median_4l_all"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_4l_all"].append(epochs)
                results_mse["mixed_4l_all"].append(train_result)
                
        ########################################### Continuous columns ###########################################   
            if TRAIN_ALL_CONTINUOS_COLUMNS:
                # train the models with all continuous columns
                X_train_mean = train_mean[continuous_columns].to_numpy()
                X_train_median = train_median[continuous_columns].to_numpy()
                X_train_mixed = train_mixed[continuous_columns].to_numpy()

                X_test_mean = test_mean[continuous_columns].to_numpy()
                X_test_median = test_median[continuous_columns].to_numpy()
                X_test_mixed = test_mixed[continuous_columns].to_numpy()

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_2l_continuous"].append(epochs)
                results_mse["mean_2l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_2l_continuous"].append(epochs)
                results_mse["median_2l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_2l_continuous"].append(epochs)
                results_mse["mixed_2l_continuous"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_3l_continuous"].append(epochs)
                results_mse["mean_3l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_3l_continuous"].append(epochs)
                results_mse["median_3l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_3l_continuous"].append(epochs)
                results_mse["mixed_3l_continuous"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_4l_continuous"].append(epochs)
                results_mse["mean_4l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_4l_continuous"].append(epochs)
                results_mse["median_4l_continuous"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_4l_continuous"].append(epochs)
                results_mse["mixed_4l_continuous"].append(train_result)

        ########################################### selected columns ###########################################
            if TRAIN_SELECTED_COLUMNS:
                # train the models with selected columns
                X_train_mean = train_mean[selected_columns].to_numpy()
                X_train_median = train_median[selected_columns].to_numpy()
                X_train_mixed = train_mixed[selected_columns].to_numpy()

                X_test_mean = test_mean[selected_columns].to_numpy()
                X_test_median = test_median[selected_columns].to_numpy()
                X_test_mixed = test_mixed[selected_columns].to_numpy()

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])
                
                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_2l_selected"].append(epochs)
                results_mse["mean_2l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_2l_selected"].append(epochs)
                results_mse["median_2l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_2l_selected"].append(epochs)
                results_mse["mixed_2l_selected"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_3l_selected"].append(epochs)
                results_mse["mean_3l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_3l_selected"].append(epochs)
                results_mse["median_3l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_3l_selected"].append(epochs)
                results_mse["mixed_3l_selected"].append(train_result)

                tf.random.set_seed(MODEL_RANDOM_SEED)
                np.random.seed(MODEL_RANDOM_SEED)
                rn.seed(MODEL_RANDOM_SEED)

                model = nn.Sequential([
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu", input_shape=[X_train_mean.shape[1]]),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="relu"),
                    nn.layers.Dense(X_train_mean.shape[1], activation="linear"),
                    nn.layers.Dense(2)
                ])

                train_result, epochs = run_cross_validation(X_train_mean, y_train, model, random_seed)
                results_epochs["mean_4l_selected"].append(epochs)
                results_mse["mean_4l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_median, y_train, model, random_seed)
                results_epochs["median_4l_selected"].append(epochs)
                results_mse["median_4l_selected"].append(train_result)
                train_result, epochs = run_cross_validation(X_train_mixed, y_train, model, random_seed)
                results_epochs["mixed_4l_selected"].append(epochs)
                results_mse["mixed_4l_selected"].append(train_result)
        
        print("CV results MSE:", json.dumps(results_mse, indent=2), sep="\n", end="\n\n")
        print("CV results epochs:", json.dumps(results_epochs, indent=2), sep="\n", end="\n\n")

        # average results
        best_mse = sys.maxsize
        best_key = ""
        for key in results_epochs.keys():
            results_epochs[key] = np.mean(results_epochs[key])
        for key in results_mse.keys():
            results_mse[key] = np.mean(results_mse[key])
            if results_mse[key] < best_mse:
                best_mse = results_mse[key]
                best_key = key
        
        print("CV average results MSE:", json.dumps(results_mse, indent=2), sep="\n", end="\n\n")
        print("CV average results epochs:", json.dumps(results_epochs, indent=2), sep="\n", end="\n\n")
        print("Best model:", f"{best_key} - epochs: {results_epochs[best_key]}, MSE: {results_mse[best_key]}", sep="\n")

    ########################################## model selected based on cross-validation results ##########################################
    else:
        train = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
        test = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")
        
        X_train = train[selected_columns].to_numpy()
        X_test = test[selected_columns].to_numpy()

        y_train = train[target_columns].to_numpy()
        y_test = test[target_columns].to_numpy()
                
        tf.random.set_seed(MODEL_RANDOM_SEED)
        np.random.seed(MODEL_RANDOM_SEED)
        rn.seed(MODEL_RANDOM_SEED)
        
        model = nn.Sequential([
                nn.layers.Dense(X_train.shape[1], activation="relu", input_shape=[X_train.shape[1]]),
                nn.layers.Dense(X_train.shape[1], activation="relu"),
                nn.layers.Dense(X_train.shape[1], activation="relu"),
                nn.layers.Dense(X_train.shape[1], activation="linear"),
                nn.layers.Dense(2)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
        test_pred, test_result = train_and_evaluate(X_train, y_train, X_test, y_test, model, 50, verbose=0)
        print(":")
        print(f"Test results of a model with 4 hidden layers trained for 50 epochs on selected columns with median value imputation:")
        print(f"  - MSE:      {metrics.mean_squared_error(y_test, test_pred)}")
        print(f"  - MAE:      {metrics.mean_absolute_error(y_test, test_pred)}")
        print(f"  - RMSE:     {np.sqrt(metrics.mean_squared_error(y_test, test_pred))}")
        print(f"  - R2 Score: {metrics.r2_score(y_test, test_pred)}", end="\n\n")

        # store predicted and ground truth values
        predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": test_pred[:, 0], "predicted PG_average_fees_(in_pounds)": test_pred[:, 1], 
                                        "truth UG_average_fees_(in_pounds)": y_test[:, 0], "truth PG_average_fees_(in_pounds)": y_test[:, 1]})
        
        print("Predicted vs. ground truth values:")
        predicted_truth.to_csv(f"results/NN_predicted_truth_seed{RANDOM_SEED}.csv", index=False)
