import os

CROSS_VALIDATION_SEED = 14
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", CROSS_VALIDATION_SEED))
# RANDOM_SEED = CROSS_VALIDATION_SEED # use our group number for the cross-validation

DATA_PATH = {
    "original":      "data/",
    "cleaning":      "data/01_cleaning/",
    "split":         "data/02_splitting/",
    "imputation":    "data/03_imputation/",
    "normalization": "data/04_normalization/",
    "numeric":       "data/05_numeric/",
}
