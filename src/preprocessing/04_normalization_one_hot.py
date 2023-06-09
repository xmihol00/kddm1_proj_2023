import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH

# file header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to


# zero mean and unit variance normalization for continuous features (apart from the target variables)
# normalized columns (continuous features):
NORMALIZED_CONTINUOUS_COLUMNS = ["UK_rank", 
                                 "World_rank", 
                                 "CWUR_score", 
                                 "Minimum_IELTS_score", 
                                 "International_students", 
                                 "Student_satisfaction", 
                                 "Estimated_cost_of_living_per_year_(in_pounds)", 
                                 "Student_enrollment_from", 
                                 "Student_enrollment_to", 
                                 "Academic_staff_from", 
                                 "Academic_staff_to",
                                 "Founded_year",
                                 "Latitude",
                                 "Longitude"]

def normalize_train(df: pd.DataFrame) -> "tuple[pd.Series, pd.Series]":
    mean = df[NORMALIZED_CONTINUOUS_COLUMNS].mean()
    std = df[NORMALIZED_CONTINUOUS_COLUMNS].std()
    df[NORMALIZED_CONTINUOUS_COLUMNS] = (df[NORMALIZED_CONTINUOUS_COLUMNS] - mean) / std

    return mean, std

def normalize_test(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> None:
    df[NORMALIZED_CONTINUOUS_COLUMNS] = (df[NORMALIZED_CONTINUOUS_COLUMNS] - mean) / std

if __name__ == "__main__":
    print("Normalization of continuous features ...")
    os.makedirs(DATA_PATH["normalization"], exist_ok=True)

    # ensure that the test set is normalized by the statistics of the training set, in order to avoid data leakage

    # normalize the mean imputed data set
    df_train = pd.read_csv(DATA_PATH["imputation"] + "Universities_train_mean.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["imputation"] + "Universities_test_mean.csv")
    df_test.set_index("Id", inplace=True)

    mean, std = normalize_train(df_train)
    normalize_test(df_test, mean, std)
    df_train.to_csv(DATA_PATH["normalization"] + "Universities_train_mean.csv")
    df_test.to_csv(DATA_PATH["normalization"] + "Universities_test_mean.csv")

    # normalize the median imputed data set
    df_train = pd.read_csv(DATA_PATH["imputation"] + "Universities_train_median.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["imputation"] + "Universities_test_median.csv")
    df_test.set_index("Id", inplace=True)

    mean, std = normalize_train(df_train)
    normalize_test(df_test, mean, std)
    df_train.to_csv(DATA_PATH["normalization"] + "Universities_train_median.csv")
    df_test.to_csv(DATA_PATH["normalization"] + "Universities_test_median.csv")

    # normalize the mixed imputed data set
    df_train = pd.read_csv(DATA_PATH["imputation"] + "Universities_train_mixed.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["imputation"] + "Universities_test_mixed.csv")
    df_test.set_index("Id", inplace=True)

    mean, std = normalize_train(df_train)
    normalize_test(df_test, mean, std)
    df_train.to_csv(DATA_PATH["normalization"] + "Universities_train_mixed.csv")
    df_test.to_csv(DATA_PATH["normalization"] + "Universities_test_mixed.csv")
