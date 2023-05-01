import numpy as np
from sklearn import metrics

# read the test predictions of all models and calculate the mean (skip the first row)
y_pred_truth_NN = np.loadtxt("results/DNN_predicted_truth.csv", delimiter=",", skiprows=1)
y_pred_truth_RF = np.loadtxt("results/RF_predicted_truth.csv", delimiter=",", skiprows=1)
# TODO: add your own model predictions here

y_pred_truth = np.mean([y_pred_truth_NN, y_pred_truth_RF], axis=0)

print("Ensemble model test results:")
print(f"  - MSE: {metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
print(f"  - RMSE: {metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2], squared=False)}")
print(f"  - MAE: {metrics.mean_absolute_error(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
print(f"  - R2 Score: {metrics.r2_score(y_pred_truth[:, 2:], y_pred_truth[:, :2])}")
