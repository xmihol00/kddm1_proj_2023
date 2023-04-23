#!/usr/bin/env bash

# pre_processing
python3 ./src/pre_processing/01_cleaning.py
python3 ./src/pre_processing/02_train_test_split.py
python3 ./src/pre_processing/03_missing_value_imputation.py
python3 ./src/pre_processing/04_normalization.py
python3 ./src/pre_processing/05_only_numeric_columns.py

# Davids Model
# python3 ./src/DNN_model.py