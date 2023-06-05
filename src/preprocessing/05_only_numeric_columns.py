import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH

def finalize_data_set(df_train: pd.DataFrame, df_test: pd.DataFrame, type: str):
    df_train.drop(columns=["Id"], inplace=True)
    df_test.drop(columns=["Id"], inplace=True)
    df_train = df_train.select_dtypes(include=["float64", "int64"])
    df_test = df_test.select_dtypes(include=["float64", "int64"])
    df_train = df_train.reindex(sorted(df_train.columns), axis=1)
    df_test = df_test.reindex(sorted(df_test.columns), axis=1)
    df_train.to_csv(DATA_PATH["numeric"] + f"Universities_train_{type}.csv", index=False)
    df_test.to_csv(DATA_PATH["numeric"] + f"Universities_test_{type}.csv", index=False)

if __name__ == "__main__":
    print("Dropping non numeric columns and the Id column...")
    os.makedirs(DATA_PATH["numeric"], exist_ok=True)

    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_mean_imputed_normalized.csv")
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_mean_imputed_normalized.csv")
    finalize_data_set(df_train, df_test, "mean")

    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_median_imputed_normalized.csv")
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_median_imputed_normalized.csv")
    finalize_data_set(df_train, df_test, "median")

    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_mixed_imputed_normalized.csv")
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_mixed_imputed_normalized.csv")
    finalize_data_set(df_train, df_test, "mixed")
