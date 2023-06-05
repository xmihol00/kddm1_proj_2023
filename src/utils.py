import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

RANDOM_SEED = 14 # our group number

DATA_PATH = {
    "original":      "data/",
    "cleaning":      "data/01_cleaning/",
    "split":         "data/02_splitting/",
    "imputation":    "data/03_imputation/",
    "normalization": "data/04_normalization/",
    "numeric":       "data/05_numeric/",
}

def get_rel_data_path():
    return [
        'data/',
        'data/01_cleaning/',
        'data/02_splitting/',
        'data/03_imputation/',
        'data/04_normalization/',
        'data/05_numeric/',
    ]

