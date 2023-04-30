#!/usr/bin/env bash

# pre_processing
python3 ./src/preprocessing/01_cleaning.py
python3 ./src/preprocessing/02_train_test_split.py
python3 ./src/preprocessing/03_missing_value_imputation.py
python3 ./src/preprocessing/04_normalization_one_hot.py
python3 ./src/preprocessing/05_only_numeric_columns.py

