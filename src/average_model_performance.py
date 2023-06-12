import pandas as pd
import argparse
import os
import sys
from sklearn import metrics
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from constants import DATASET_RANDOM_SEEDS

parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str, help="Start of file names to be averaged.")
parser.add_argument("model_name", type=str, help="Name of the model, which is averaged.")
args = parser.parse_args()

def average_model_performance(file_name: str):
    file_name += "_seed"
    MSEs = []
    MAEs = []
    RMSEs = []
    R2s = []

    for random_seed in DATASET_RANDOM_SEEDS:
        predicted_truth = np.loadtxt(os.path.join("results", f"{file_name}{random_seed}.csv"), delimiter=",", skiprows=1)
        MSEs.append(metrics.mean_squared_error(predicted_truth[:, 2:], predicted_truth[:, :2]))
        MAEs.append(metrics.mean_absolute_error(predicted_truth[:, 2:], predicted_truth[:, :2]))
        RMSEs.append(np.sqrt(metrics.mean_squared_error(predicted_truth[:, 2:], predicted_truth[:, :2])))
        R2s.append(metrics.r2_score(predicted_truth[:, 2:], predicted_truth[:, :2]))
    
    return np.mean(MSEs), np.mean(MAEs), np.mean(RMSEs), np.mean(R2s)

if __name__ == "__main__":
    MSE, MAE, RMSE, R2 = average_model_performance(args.file_name)
    print(f"{args.model_name} average performance across random seeds {DATASET_RANDOM_SEEDS[0]}-{DATASET_RANDOM_SEEDS[-1]}:")
    print(f"  - MSE: {MSE}")
    print(f"  - MAE: {MAE}")
    print(f"  - RMSE: {RMSE}")
    print(f"  - R2 Score: {R2}")
