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
from seed import RANDOM_SEED

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

def run_cross_validation(optimizer, X_train, y_train, model, verbose=0):
    cv_results = []
    cv_epochs = []
    patience = 100
    
    for train_index, test_index in ms.KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        
        # train the model on each fold
        model.compile(optimizer=optimizer, loss='mse')
        history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), epochs=10000, batch_size=16, verbose=verbose, 
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

    return result, epochs


def train_and_evaluate(X_train, y_train, X_test, y_test, model, optimizer, epochs, verbose=0):
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=verbose)

    # evaluate the model on the test set
    pred_result = model.predict(X_test, verbose=verbose)
    eval_result = model.evaluate(X_test, y_test, verbose=verbose)
    if verbose:
        print(f"Results of a model with all columns.", f"  - loss: {eval_result}", 
              f"  - test samples (predicted, ground truth):", np.stack((pred_result, y_test)).T[:10], sep="\n")
    
    return pred_result, eval_result

if __name__ == "__main__":
    train_mean = pd.read_csv("data/Universities_train_mean.csv")
    train_median = pd.read_csv("data/Universities_train_median.csv")

    test_mean = pd.read_csv("data/Universities_test_mean.csv")
    test_median = pd.read_csv("data/Universities_test_median.csv")

    # separate data frame to X and y
    y_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
    X_train_mean = train_mean.drop(columns=y_columns).to_numpy()
    y_train_mean = train_mean[y_columns].to_numpy()
    X_train_median = train_median.drop(columns=y_columns).to_numpy()
    y_train_median = train_median[y_columns].to_numpy()

    X_test_mean = test_mean.drop(columns=y_columns).to_numpy()
    y_test_mean = test_mean[y_columns].to_numpy()
    X_test_median = test_median.drop(columns=y_columns).to_numpy()
    y_test_median = test_median[y_columns].to_numpy()

    results = []

########################################### All columns ###########################################

    model = nn.Sequential([
        nn.layers.Dense(2, activation='linear', input_shape=[X_train_mean.shape[1]])
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - linear model on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - linear model on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on non-categorical with median value imputation columns average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    results.append("")

########################################### Non-categorical columns ###########################################   

    selected_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                        "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                        "International_students", "Academic_staff_from", "Academic_staff_to"]

    X_train_mean = train_mean[selected_columns].to_numpy()
    y_train_mean = train_mean[y_columns].to_numpy()
    X_train_median = train_median[selected_columns].to_numpy()
    y_train_median = train_median[y_columns].to_numpy()

    X_test_mean = test_mean[selected_columns].to_numpy()
    y_test_mean = test_mean[y_columns].to_numpy()
    X_test_median = test_median[selected_columns].to_numpy()
    y_test_median = test_median[y_columns].to_numpy()

    model = nn.Sequential([
        nn.layers.Dense(2, activation='linear', input_shape=[X_train_mean.shape[1]])
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - linear model on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - linear model on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on non-categorical columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on non-categorical columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    results.append("")

########################################### selected columns ###########################################

    selected_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "International_students", "Minimum_IELTS_score",
                        "Student_satisfaction", "UK_rank", "World_rank"]

    X_train_mean = train_mean[selected_columns].to_numpy()
    y_train_mean = train_mean[y_columns].to_numpy()
    X_train_median = train_median[selected_columns].to_numpy()
    y_train_median = train_median[y_columns].to_numpy()

    X_test_mean = test_mean[selected_columns].to_numpy()
    y_test_mean = test_mean[y_columns].to_numpy()
    X_test_median = test_median[selected_columns].to_numpy()
    y_test_median = test_median[y_columns].to_numpy()

    model = nn.Sequential([
        nn.layers.Dense(2, activation='linear', input_shape=[X_train_mean.shape[1]])
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.4)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - linear model on selected columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.4)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - linear model on selected columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on selected columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 1 hidden layer on selected columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on selected columns with mean value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 2 hidden layers on selected columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")

    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_mean, y_train_mean, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on selected columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_result, epochs = run_cross_validation(optimizer, X_train_median, y_train_median, model, verbose=0)
    results.append(f"  - non-linear model with 3 hidden layers on selected columns with median value imputation average validation MSE: {train_result}, with average epochs: {epochs}")
    results.append("")

    print("Cross-validation results:")
    for result in results:
        print(result)

########################################## model selected based on cross-validation ##########################################

    selected_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                        "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to", 
                        "International_students", "Academic_staff_from", "Academic_staff_to"]

    X_train_mean = train_mean[selected_columns].to_numpy()
    y_train_mean = train_mean[y_columns].to_numpy()
    X_train_median = train_median[selected_columns].to_numpy()
    y_train_median = train_median[y_columns].to_numpy()

    X_test_mean = test_mean[selected_columns].to_numpy()
    y_test_mean = test_mean[y_columns].to_numpy()
    X_test_median = test_median[selected_columns].to_numpy()
    y_test_median = test_median[y_columns].to_numpy()
    
    model = nn.Sequential([
        nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
        nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
        nn.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    test_pred, test_result = train_and_evaluate(X_train_median, y_train_median, X_test_median, y_test_median, model, optimizer, 14, verbose=0)
    print("Test results:")
    print(f"  - non-linear model with 3 hidden layer trained for 14 epochs on non-categorical columns with median value imputation test MSE: {metrics.mean_squared_error(y_test_mean, test_pred)}")
    print(f"  - non-linear model with 3 hidden layer trained for 14 epochs on non-categorical columns with median value imputation test RMSE: {metrics.mean_squared_error(y_test_mean, test_pred, squared=False)}")
    print(f"  - non-linear model with 3 hidden layer trained for 14 epochs on non-categorical columns with median value imputation test MAE: {metrics.mean_absolute_error(y_test_mean, test_pred)}", end="\n\n")

    # crate a data frame with the results of the selected model
    predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": test_pred[:, 0], "predicted PG_average_fees_(in_pounds)": test_pred[:, 1], 
                                    "truth UG_average_fees_(in_pounds)": y_test_median[:, 0], "truth PG_average_fees_(in_pounds)": y_test_median[:, 1]})
    
    print("Predicted vs. ground truth values:")
    print(predicted_truth)
    predicted_truth.to_csv("models/DNN_predicted_truth.csv")
