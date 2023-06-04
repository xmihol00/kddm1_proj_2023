import os, sys
from pathlib import Path

import pandas as pd
import numpy as np

os.chdir(Path(__file__).parents[2])
sys.path.append(os.getcwd())
from src.utils import get_rel_data_path

DATA_PATH = get_rel_data_path()
os.makedirs(DATA_PATH[5], exist_ok=True)

################################################################################
def analyzeDroppedColumns(df: pd.DataFrame):
  # Dropped non numeric columns ("object")
  print('\n' + '#'*120)
  print("Dropped non numeric columns:")
  print(df.select_dtypes(exclude='number').columns)

  # Dropped columns contain NaN
  print('\n' + '#'*120)
  print("Dropped null rows:")
  
  nan_col_count = df.isna().sum()
  nan_row_count = df.isna().sum(axis=1)
  for i, x in enumerate(nan_col_count):
    if x != 0:
      print("{:5} {:50} {:3}".format(i, df.columns[i], x))
  print('-'*120)
  print("nr Null columns:    {:40}".format(np.sum(nan_col_count != 0)))
  print("nr Null rows:       {:40}".format(np.sum(nan_row_count != 0).sum()))
  print("sum Null values:    {:40}".format(nan_col_count.sum()))


################################################################################
print("Dropping non numeric columns, Id column and columns with NaNs...")


################################################################################
# read
universities_train_mean = pd.read_csv(DATA_PATH[4] + "Universities_train_mean_imputed_normalized.csv")
universities_train_median = pd.read_csv(DATA_PATH[4] + "Universities_train_median_imputed_normalized.csv")
universities_train_mixed = pd.read_csv(DATA_PATH[4] + "Universities_train_mixed_imputed_normalized.csv")

universities_test_mean = pd.read_csv(DATA_PATH[4] + "Universities_test_mean_imputed_normalized.csv")
universities_test_median = pd.read_csv(DATA_PATH[4] + "Universities_test_median_imputed_normalized.csv")
universities_test_mixed = pd.read_csv(DATA_PATH[4] + "Universities_test_mixed_imputed_normalized.csv")


################################################################################
# analyze
# analyzeDroppedColumns(universities_train_mean)

################################################################################
# drop non numeric columns, columns with NaNs and the Id column
universities_train_mean = universities_train_mean.select_dtypes(include=['float64', 'int64'])
universities_train_median = universities_train_median.select_dtypes(include=['float64', 'int64'])
universities_train_mean.dropna(axis=1, inplace=True)
universities_train_median.dropna(axis=1, inplace=True)
universities_train_mean.drop(columns=["Id"], inplace=True)
universities_train_median.drop(columns=["Id"], inplace=True)

universities_test_mean = universities_test_mean.select_dtypes(include=['float64', 'int64'])
universities_test_median = universities_test_median.select_dtypes(include=['float64', 'int64'])
universities_test_mean.dropna(axis=1, inplace=True)
universities_test_median.dropna(axis=1, inplace=True)
universities_test_mean.drop(columns=["Id"], inplace=True)
universities_test_median.drop(columns=["Id"], inplace=True)

# save the data to final train and test files
universities_train_mean.to_csv(DATA_PATH[5] + "Universities_train_mean.csv", index=False)
universities_train_median.to_csv(DATA_PATH[5] + "Universities_train_median.csv", index=False)
universities_train_mixed.to_csv(DATA_PATH[5] + "Universities_train_mixed.csv", index=False)

universities_test_mean.to_csv(DATA_PATH[5] + "Universities_test_mean.csv", index=False)
universities_test_median.to_csv(DATA_PATH[5] + "Universities_test_median.csv", index=False)
universities_test_mixed.to_csv(DATA_PATH[5] + "Universities_test_mixed.csv", index=False)
