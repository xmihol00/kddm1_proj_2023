import os, sys, argparse
import pandas as pd
from enum import Enum
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH, CV_SPLITS, RANDOM_SEED, DATASET_RANDOM_SEEDS

parser = argparse.ArgumentParser()
parser.add_argument("-cv", "--cross_validation", action="store_true", help="Perform cross-validation.")
parser.add_argument("-ds", "--dataset_seed", type=int, default=RANDOM_SEED, help="Random seed of the data set splitting.")
args = parser.parse_args()
TRAIN_MODEL = True;

class PreProcessing(Enum):
    MEAN = 1
    MEDIAN = 2
    MIXED = 3

def getBestParams(X_train, y_train):
    svr = SVR()
    param_grid = {
        'C' : [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5],
        'kernel' : ['linear', 'rbf'],
        'degree' : [2, 3, 4],
        'epsilon' : [0.0001, 0.0005, 0.005, 0.01, 0.25, 0.5, 1, 5],
        'gamma' : [0.0001, 0.001, 0.01, 0.1, 1]
    }

    gs = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=CV_SPLITS)
    gs.fit(X_train, y_train)

    c = gs.best_params_['C']
    degree = gs.best_params_['degree']
    epsilon = gs.best_params_['epsilon']
    kernel = gs.best_params_['kernel']
    gamma = gs.best_params_['gamma']

    return c, degree, epsilon, kernel, gamma

def getPerformance(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    r2 = metrics.r2_score(y_true, y_pred)
    return mse, mae, rmse, r2

def performSVR(X_train : pd.DataFrame, y_train : pd.DataFrame, X_test : pd.DataFrame,  y_test : pd.DataFrame, targets: "list[str]"):
    y_test_result = []
    y_pred = []
    mse = []
    mae = []
    rmse = []
    r2 = []
    c = []
    degree = []
    epsilon = []
    kernel = []
    gamma = []

    for target in targets:
        if (args.cross_validation):
            c_target, degree_target, epsilon_target, kernel_target, gamma_target = getBestParams(X_train, y_train[target].to_numpy())
        else:
            gamma_target = 0.0001
            c_target = 5
            degree_target = 2
            epsilon_target = 5.0
            kernel_target = 'linear'

        svr = SVR(kernel=kernel_target, degree=degree_target, epsilon=epsilon_target, C=c_target, gamma=gamma_target)
        c.append(c_target)
        degree.append(degree_target)
        epsilon.append(epsilon_target)
        kernel.append(kernel_target)
        gamma.append(gamma_target)

        svr.fit(X_train, y_train[target].to_numpy())
        test = y_test[target].to_numpy()
        pred = svr.predict(X_test)
        y_pred.append(pred)
        y_test_result.append(test)

        mse_target, mae_target, rmse_target, r2_target = getPerformance(test, pred)
        mse.append(mse_target)
        mae.append(mae_target)
        rmse.append(rmse_target)
        r2.append(r2_target)

    return y_test_result, y_pred, mse, mae, rmse, r2, c, degree, kernel, epsilon, gamma

if __name__ == "__main__":
    target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]

    best_UG = pd.DataFrame(columns=['seed', 'target', 'features', 'imputation', 'mse', 'mae', 'rmse', 'r2', 'C', 'degree', 'kernel', 'epsilon', 'gamma'])
    best_PG = pd.DataFrame(columns=['seed', 'target', 'features', 'imputation', 'mse', 'mae', 'rmse', 'r2', 'C', 'degree', 'kernel', 'epsilon', 'gamma'])

    seed = []
    target = []
    features = []
    imputation = []
    mse = []
    mae = []
    rmse = []
    r2 = []
    c = []
    degree = []
    kernel = []
    epsilon = []
    gamma = []

    y_pred = []
    y_test = []

    train_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mean.csv")
    test_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mean.csv")

    train_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
    test_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")

    train_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mixed.csv")
    test_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mixed.csv")

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

            all_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score", "Latitude" , "Longitude",
                                        "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to",
                                        "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]

            # separate data frame to X and y
            X_train_mean = train_mean[all_columns].to_numpy()
            y_train_mean = train_mean[target_columns]
            X_train_median = train_median[all_columns].to_numpy()
            y_train_median = train_median[target_columns]
            X_train_mixed = train_mixed[all_columns].to_numpy()
            y_train_mixed = train_mixed[target_columns]

            X_test_mean = test_mean[all_columns].to_numpy()
            y_test_mean = test_mean[target_columns]
            X_test_median = test_median[all_columns].to_numpy()
            y_test_median = test_median[target_columns]
            X_test_mixed = test_mixed[all_columns].to_numpy()
            y_test_mixed = test_mixed[target_columns]

            print(f"Cross validation results:")

            feature_selection = "all columns"

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur = performSVR(X_train_mean, y_train_mean, X_test_mean, y_test_mean, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mean", "mean"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur = performSVR(X_train_median, y_train_median, X_test_median, y_test_median, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["median", "median"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_mixed, y_train_mixed, X_test_mixed, y_test_mixed, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mixed", "mixed"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            feature_selection = "continuous columns"

            continuous_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score",
                                "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to",
                                "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]

            X_train_mean = train_mean[continuous_columns].to_numpy()
            y_train_mean = train_mean[target_columns]
            X_train_median = train_median[continuous_columns].to_numpy()
            y_train_median = train_median[target_columns]
            X_train_mixed = train_mixed[continuous_columns].to_numpy()
            y_train_mixed = train_mixed[target_columns]

            X_test_mean = test_mean[continuous_columns].to_numpy()
            y_test_mean = test_mean[target_columns]
            X_test_median = test_median[continuous_columns].to_numpy()
            y_test_median = test_median[target_columns]
            X_test_mixed = test_mixed[continuous_columns].to_numpy()
            y_test_mixed = test_mixed[target_columns]

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_mean, y_train_mean, X_test_mean, y_test_mean, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mean", "mean"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_median, y_train_median, X_test_median, y_test_median, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["median", "median"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_mixed, y_train_mixed, X_test_mixed, y_test_mixed, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mixed", "mixed"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            feature_selection = "selected columns"

            selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students",
                                "Academic_staff_from", "Academic_staff_to"]

            X_train_mean = train_mean[selected_columns].to_numpy()
            y_train_mean = train_mean[target_columns]
            X_train_median = train_median[selected_columns].to_numpy()
            y_train_median = train_median[target_columns]
            X_train_mixed = train_mixed[selected_columns].to_numpy()
            y_train_mixed = train_mixed[target_columns]

            X_test_mean = test_mean[selected_columns].to_numpy()
            y_test_mean = test_mean[target_columns]
            X_test_median = test_median[selected_columns].to_numpy()
            y_test_median = test_median[target_columns]
            X_test_mixed = test_mixed[selected_columns].to_numpy()
            y_test_mixed = test_mixed[target_columns]

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_mean, y_train_mean, X_test_mean, y_test_mean, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mean", "mean"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_median, y_train_median, X_test_median, y_test_median, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["median", "median"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur  = performSVR(X_train_mixed, y_train_mixed, X_test_mixed, y_test_mixed, target_columns)
            seed.extend([random_seed, random_seed])
            target.extend(target_columns)
            features.extend([feature_selection, feature_selection])
            imputation.extend(["mixed", "mixed"])
            mse.extend(mse_cur)
            mae.extend(mae_cur)
            rmse.extend(rmse_cur)
            r2.extend(r2_cur)
            c.extend(c_cur)
            degree.extend(degree_cur)
            kernel.extend(kernel_cur)
            epsilon.extend(epsilon_cur)
            gamma.extend(gamma_cur)

            result = pd.DataFrame({
                'seed' : seed,
                'target' : target,
                'features' : features,
                'imputation' : imputation,
                'mse' : mse,
                'mae' : mae,
                'rmse' : rmse,
                'r2' : r2,
                'C' : c,
                'degree' : degree,
                'kernel' : kernel,
                'epsilon' : epsilon,
                'gamma' : gamma
            })

            current_frame = result[result['seed'] == random_seed]
            print(current_frame)
            current_frame_UG = current_frame[current_frame['target'] == 'UG_average_fees_(in_pounds)']
            current_frame_PG = current_frame[current_frame['target'] == 'PG_average_fees_(in_pounds)']
            best_UG = pd.concat([best_UG, current_frame_UG[current_frame_UG['mse'] == current_frame_UG['mse'].min()]])
            best_PG = pd.concat([best_PG, current_frame_PG[current_frame_PG['mse'] == current_frame_PG['mse'].min()]])

    else:
        y_test.clear()
        y_pred.clear()

        feature_selection = "selected columns"

        selected_columns = ["UK_rank", "World_rank", "CWUR_score", "Minimum_IELTS_score", "International_students",
                            "Academic_staff_from", "Academic_staff_to"]


        #feature_selection = "all columns"

        #all_columns = ["CWUR_score", "Estimated_cost_of_living_per_year_(in_pounds)", "Minimum_IELTS_score", "Latitude" , "Longitude",
        #                                "Student_satisfaction", "UK_rank", "World_rank", "Student_enrollment_from", "Student_enrollment_to",
        #                                "International_students", "Academic_staff_from", "Academic_staff_to", "Founded_year"]

        X_train_mixed = train_mixed[selected_columns].to_numpy()
        y_train_mixed = train_mixed[target_columns]
        X_test_mixed = test_mixed[selected_columns].to_numpy()
        y_test_mixed = test_mixed[target_columns]

        y_test_result, y_pred, mse_cur, mae_cur, rmse_cur, r2_cur, c_cur, degree_cur, kernel_cur, epsilon_cur, gamma_cur = performSVR(X_train_mixed, y_train_mixed, X_test_mixed, y_test_mixed, target_columns)
        seed.extend([RANDOM_SEED, RANDOM_SEED])
        target.extend(target_columns)
        features.extend([feature_selection, feature_selection])
        imputation.extend(["mixed", "mixed"])
        mse.extend(mse_cur)
        mae.extend(mae_cur)
        rmse.extend(rmse_cur)
        r2.extend(r2_cur)
        c.extend(c_cur)
        degree.extend(degree_cur)
        kernel.extend(kernel_cur)
        epsilon.extend(epsilon_cur)
        gamma.extend(gamma_cur)

        result = pd.DataFrame({
                'seed' : seed,
                'target' : target,
                'features' : features,
                'imputation' : imputation,
                'mse' : mse,
                'mae' : mae,
                'rmse' : rmse,
                'r2' : r2,
                'C' : c,
                'degree' : degree,
                'kernel' : kernel,
                'epsilon' : epsilon,
                'gamma' : gamma
            })

        y_test = y_test_result

        current_frame = result[result['seed'] == RANDOM_SEED]
        print("Test result:")
        print(current_frame)
        current_frame_UG = current_frame[current_frame['target'] == 'UG_average_fees_(in_pounds)']
        current_frame_PG = current_frame[current_frame['target'] == 'PG_average_fees_(in_pounds)']
        best_UG = pd.concat([best_UG, current_frame_UG[current_frame_UG['mse'] == current_frame_UG['mse'].min()]])
        best_PG = pd.concat([best_PG, current_frame_PG[current_frame_PG['mse'] == current_frame_PG['mse'].min()]])
        in_total = pd.concat([best_UG, best_PG])
        print("\n    Avg. MSE: {:0.4f} - Avg. RMSE: {:0.4f} - Avg. MAE: {:0.4f} - Avg. R^2: {:0.4f}\n".format(in_total['mse'].median(), in_total['rmse'].median(), in_total['mae'].median(), in_total['r2'].median()))

    if (args.cross_validation):
        print('Best performing parameters:')
        print(' UG_average_fees_(in_pounds):')
        print('    Features: {0}, imputation: {1}, C: {2}, degree: {3}, kernel: {4}, epsilon: {5}, gamma: {6}'.format(best_UG['features'].mode()[0], best_UG['imputation'].mode()[0], best_UG['C'].mode()[0], best_UG['degree'].mode()[0], best_UG['kernel'].mode()[0], best_UG['epsilon'].mode()[0], best_UG['gamma'].mode()[0]))
        print("    Avg. MSE: {:0.4f} - Avg. RMSE: {:0.4f} - Avg. MAE: {:0.4f} - Avg. R^2: {:0.4f}".format(best_UG['mse'].median(), best_UG['rmse'].median(), best_UG['mae'].median(), best_UG['r2'].median()))
        print(' PG_average_fees_(in_pounds):')
        print('    Features: {0}, imputation: {1}, C: {2}, degree: {3}, kernel: {4}, epsilon: {5}, gamma: {6}'.format(best_PG['features'].mode()[0], best_PG['imputation'].mode()[0], best_PG['C'].mode()[0], best_PG['degree'].mode()[0], best_PG['kernel'].mode()[0], best_PG['epsilon'].mode()[0], best_PG['gamma'].mode()[0]))
        print("    Avg. MSE: {:0.4f} - Avg. RMSE: {:0.4f} - Avg. MAE: {:0.4f} - Avg. R^2: {:0.4f}".format(best_PG['mse'].median(), best_PG['rmse'].median(), best_PG['mae'].median(), best_PG['r2'].median()))
        in_total = pd.concat([best_UG, best_PG])
        print('Best performing parameters (in total):')
        print(' Features: {0}, imputation: {1}, C: {2}, degree: {3}, kernel: {4}, epsilon: {5}, gamma: {6}'.format(in_total['features'].mode()[0], in_total['imputation'].mode()[0], in_total['C'].mode()[0], in_total['degree'].mode()[0], in_total['kernel'].mode()[0], in_total['epsilon'].mode()[0], in_total['gamma'].mode()[0]))
    else:
        predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[0],
                                        "predicted PG_average_fees_(in_pounds)": y_pred[1],
                                        "truth UG_average_fees_(in_pounds)": y_test[0],
                                        "truth PG_average_fees_(in_pounds)": y_test[1]})

        print("Predicted vs. ground truth values:")
        print(predicted_truth)

        predicted_truth.to_csv(f"results/SVR_predicted_truth_seed{args.dataset_seed}.csv", index=False)
