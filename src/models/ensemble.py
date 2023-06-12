import os
import sys
import numpy as np
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATASET_RANDOM_SEEDS

models = [
    "baseline",
    "NN", 
    "RF", 
    "SVR"
]
MSEs = []
MAEs = []
RMSEs = []
R2s = []

for i in DATASET_RANDOM_SEEDS:
    y_preds_truths = []
    for model in models:
        y_preds_truths.append(np.loadtxt(os.path.join("results", f"{model}_predicted_truth_seed{i}.csv"), delimiter=",", skiprows=1))
    
    y_pred_truth = np.mean(y_preds_truths, axis=0)
    MSEs.append(metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2]))
    MAEs.append(metrics.mean_absolute_error(y_pred_truth[:, 2:], y_pred_truth[:, :2]))
    RMSEs.append(np.sqrt(metrics.mean_squared_error(y_pred_truth[:, 2:], y_pred_truth[:, :2])))
    R2s.append(metrics.r2_score(y_pred_truth[:, 2:], y_pred_truth[:, :2]))

print("Ensemble model test results:")
print(f"  - MSE:      {np.median(MSEs)}")
print(f"  - MAE:      {np.median(MAEs)}")
print(f"  - RMSE:     {np.median(RMSEs)}")
print(f"  - R2 Score: {np.median(R2s)}")
