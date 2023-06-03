import os
from pathlib import Path
import numpy as np
from sklearn import metrics

os.chdir(Path(__file__).parents[2])

# read the test predictions of all models and calculate the mean (skip the first row)
y_pred_truth_NN = np.loadtxt("results/DNN_predicted_truth.csv", delimiter=",", skiprows=1)
y_pred_truth_RF = np.loadtxt("results/RF_predicted_truth.csv", delimiter=",", skiprows=1)
y_pred_truth_SVR = np.loadtxt("results/SVR_predicted_truth.csv", delimiter=",", skiprows=1)

y_pred_truth = np.mean([y_pred_truth_NN, y_pred_truth_RF, y_pred_truth_SVR], axis=0)

print("Ensemble model test results:")
print(f"  - MSE: {metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
print(f"  - RMSE: {metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2], squared=False)}")
print(f"  - MAE: {metrics.mean_absolute_error(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
print(f"  - R2 Score: {metrics.r2_score(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
