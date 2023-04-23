#!/usr/bin/env bash


python3 ./src/cleaning.py
python3 ./src/train_test_split.py
python3 ./src/missing_value_imputation.py
python3 ./src/only_numeric_columns.py

# Davids Model
python3 ./src/DNN_model.py