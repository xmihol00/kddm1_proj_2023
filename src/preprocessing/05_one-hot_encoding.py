import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH

# one-hot encoding for categorical features
# one-hot encoded columns (categorical features):
ONE_HOT_ENCODED_COLUMNS = ["Region",
                           "Control_type",
                           "Academic_Calender",
                           "Campus_setting"]

def one_hot_encode(df_train: pd.DataFrame, df_test):
    df_train = pd.get_dummies(df_train, columns=ONE_HOT_ENCODED_COLUMNS)
    df_test = pd.get_dummies(df_test, columns=ONE_HOT_ENCODED_COLUMNS)

    # make sure that the one-hot encoded columns are the same for both data frames
    missing_test_columns = set(df_train.columns) - set(df_test.columns)
    for column in missing_test_columns:
        df_test[column] = 0
    
    missing_train_columns = set(df_test.columns) - set(df_train.columns)
    for column in missing_train_columns:
        df_train[column] = 0
    
    return df_train, df_test


if __name__ == "__main__":
    print("One-hot encoding categorical features ...")
    os.makedirs(DATA_PATH["one-hot"], exist_ok=True)


    # normalize the mean imputed data set
    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_mean.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_mean.csv")
    df_test.set_index("Id", inplace=True)

    df_train, df_test = one_hot_encode(df_train, df_test)
    df_train.to_csv(DATA_PATH["one-hot"] + "Universities_train_mean.csv")
    df_test.to_csv(DATA_PATH["one-hot"] + "Universities_test_mean.csv")

    # normalize the median imputed data set
    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_median.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_median.csv")
    df_test.set_index("Id", inplace=True)

    df_train, df_test = one_hot_encode(df_train, df_test)
    df_train.to_csv(DATA_PATH["one-hot"] + "Universities_train_median.csv")
    df_test.to_csv(DATA_PATH["one-hot"] + "Universities_test_median.csv")

    # normalize the mixed imputed data set
    df_train = pd.read_csv(DATA_PATH["normalization"] + "Universities_train_mixed.csv")
    df_train.set_index("Id", inplace=True)
    df_test = pd.read_csv(DATA_PATH["normalization"] + "Universities_test_mixed.csv")
    df_test.set_index("Id", inplace=True)

    df_train, df_test = one_hot_encode(df_train, df_test)
    df_train.to_csv(DATA_PATH["one-hot"] + "Universities_train_mixed.csv")
    df_test.to_csv(DATA_PATH["one-hot"] + "Universities_test_mixed.csv")
