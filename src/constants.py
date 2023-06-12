import os

CROSS_VALIDATION_SEED = 14 # TODO: remove
MODEL_RANDOM_SEED = 42
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", CROSS_VALIDATION_SEED))
CV_SPLITS = 5

DATASET_RANDOM_SEEDS = list(range(40, 50))

DATA_PATH = {
    "original":      "data/",
    "cleaning":      "data/01_cleaning/",
    "split":         "data/02_splitting/",
    "imputation":    "data/03_imputation/",
    "normalization": "data/04_normalization/",
    "one-hot":       "data/05_one-hot/",
    "numeric":       "data/06_numeric/",
}
