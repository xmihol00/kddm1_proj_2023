import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # ensure includes of our files work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH

# file header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to

def impute_train_mean(df: pd.DataFrame, columns: "list[str]") -> "dict[str, float]":
    column_mean_dict = {}
    for col in columns: 
        mean = df[col].mean()
        column_mean_dict[col] = mean
        df[col].fillna(mean, inplace=True)
    
    return column_mean_dict

def impute_test_mean(df: pd.DataFrame, column_mean_dict: "dict[str, float]") -> None:
    for col, mean in column_mean_dict.items(): 
        df[col].fillna(mean, inplace=True)

def impute_train_median(df: pd.DataFrame, columns: "list[str]") -> "dict[str, float]":
    column_median_dict = {}
    for col in columns: 
        median = df[col].median()
        column_median_dict[col] = median
        df[col].fillna(median, inplace=True)
    
    return column_median_dict

def impute_test_median(df: pd.DataFrame, column_median_dict: "dict[str, float]") -> None:
    for col, median in column_median_dict.items(): 
        df[col].fillna(median, inplace=True)

def impute_train_mode(df: pd.DataFrame, columns: "list[str]") -> "dict[str, float]":
    column_mode_dict = {}
    for col in columns: 
        mode = df[col].mode()[0]
        column_mode_dict[col] = mode
        df[col].fillna(mode, inplace=True)

    return column_mode_dict

def impute_test_mode(df: pd.DataFrame, column_mode_dict: "dict[str, float]") -> None:
    for col, mode in column_mode_dict.items(): 
        df[col].fillna(mode, inplace=True)

def impute_constant(df: pd.DataFrame, columns: "list[str]", constant: float) -> None:
    for col in columns: 
        df[col].fillna(constant, inplace=True)

def impute_founded_year(df : pd.DataFrame):
    founded_years = {
        "University of Oxford": 1096,
        "University of St Andrews": 1413,
        "Lancaster University": 1964,
        "University of Warwick": 1965,
        "University of Bath": 1966,
        "University of Leeds": 1904,
        "University of Birmingham": 1900,
        "Harper Adams University": 1901,
        "University of Southampton": 1862,
        "Newcastle University": 1963,
        "King's College London": 1829,
        "Swansea University": 1920,
        "University of Essex": 1964,
        "University of Sussex": 1961,
        "Aberystwyth University": 1872,
        "Aston University": 1895,
        "Edge Hill University": 1882,
        "Liverpool Hope University": 1844,
        "De Montfort University": 1870,
        "St Mary's University, Twickenham": 1850,
        "University of Winchester": 1840,
        "Abertay University": 1994,
        "University of Bradford": 1832,
        "Buckinghamshire New University": 1891,
        "Newman University": 1968,
        "University of Northampton": 1999,
        "University of Cumbria": 1822,
        "University of Bedfordshire": 1882
    }
    
    nan_indices = df[df["Founded_year"].isnull()].index
    for nan_index in nan_indices:
        university_name = df.loc[nan_index, "University_name"]
        df.loc[nan_index, "Founded_year"] = founded_years[university_name]

def impute_train_CWUR_score(df: pd.DataFrame) -> "tuple[float, float]":
    lm = LinearRegression()
    df_regression = df[df["CWUR_score"].notna()]
    lm.fit(df_regression["World_rank"].to_numpy().reshape(-1, 1), df_regression["CWUR_score"])
    slope = lm.coef_[0]
    intercept = lm.intercept_

    # get indices of rows with missing CWUR_score
    nan_indices = df[df["CWUR_score"].isnull()].index
    for nan_index in nan_indices:
        df.loc[nan_index, "CWUR_score"] = df.loc[nan_index, "World_rank"] * slope + intercept
    
    return slope, intercept

def impute_test_CWUR_score(df: pd.DataFrame, slope: float, intercept: float) -> None:
    nan_indices = df[df["CWUR_score"].isnull()].index
    for nan_index in nan_indices:
        df.loc[nan_index, "CWUR_score"] = df.loc[nan_index, "World_rank"] * slope + intercept

def impute_train_student_satisfaction(df: pd.DataFrame) -> float:
    median = df["Student_satisfaction"].median()
    df["Student_satisfaction"].fillna(median, inplace=True)
    return median

def impute_test_student_satisfaction(df: pd.DataFrame, median: float) -> None:
    df["Student_satisfaction"].fillna(median, inplace=True)

def impute_train_academic_calender(df: pd.DataFrame) -> float:
    mode = df["Academic_Calender"].value_counts().idxmax()
    df["Academic_Calender"] = df["Academic_Calender"].fillna(mode)
    return mode

def impute_test_academic_calender(df: pd.DataFrame, mode: float) -> None:
    df["Academic_Calender"] = df["Academic_Calender"].fillna(mode)


def impute_train_campus_setting(df : pd.DataFrame) -> KNeighborsClassifier:
    df_train = df[["Latitude", "Longitude", "Campus_setting"]].copy()

    df_train = df_train[df_train["Campus_setting"].notna()] # remove rows with missing Campus_setting
    train_X = df_train.drop(columns="Campus_setting")
    train_y = df_train["Campus_setting"]

    KNN_classifier = KNeighborsClassifier(n_neighbors=4, p=2)
    KNN_classifier.fit(train_X, train_y)

    # predict missing Campus_setting and insert into original data frame
    df_test = df[["Latitude", "Longitude", "Campus_setting"]].copy()
    df_test = df_test[df["Campus_setting"].isnull()]
    test_X = df_test.drop(columns="Campus_setting")
    test_y = KNN_classifier.predict(test_X)
    df.loc[df["Campus_setting"].isnull(), "Campus_setting"] = test_y

    return KNN_classifier

def impute_test_campus_setting(df : pd.DataFrame, KNN_classifier: KNeighborsClassifier) -> None:
    df_test = df[["Latitude", "Longitude", "Campus_setting"]].copy()
    df_test = df_test[df["Campus_setting"].isnull()]
    test_X = df_test.drop(columns="Campus_setting")
    test_y = KNN_classifier.predict(test_X)
    df.loc[df["Campus_setting"].isnull(), "Campus_setting"] = test_y

def mean_imputation(df_train: pd.DataFrame, df_test: pd.DataFrame, dst_path: str) -> None:
    # ensure input data frame is not modified
    df_train_mean_imputed = df_train.copy()
    df_test_mean_imputed = df_test.copy()

    column_mean_dict = impute_train_mean(df_train_mean_imputed, ["CWUR_score", "Student_satisfaction", "Founded_year"])
    # categorical columns must be imputed with mode
    column_mode_dict = impute_train_mode(df_train_mean_imputed, ["Academic_Calender", "Campus_setting"])

    # impute test data with the mean and mode of the training data, so that the test data is not leaked into the training data
    impute_test_mean(df_test_mean_imputed, column_mean_dict)
    impute_test_mode(df_test_mean_imputed, column_mode_dict)

    impute_constant(df_train_mean_imputed, ["Academic_staff_from"], 5_000)
    impute_constant(df_test_mean_imputed, ["Academic_staff_from"], 5_000)

    impute_constant(df_train_mean_imputed, ["Academic_staff_to"], 10_000)
    impute_constant(df_test_mean_imputed, ["Academic_staff_to"], 10_000)

    # drop university name, not needed anymore
    df_train_mean_imputed.drop(columns="University_name", inplace=True)
    df_test_mean_imputed.drop(columns="University_name", inplace=True)

    # save
    df_train_mean_imputed.to_csv(dst_path + "Universities_train_mean.csv")
    df_test_mean_imputed.to_csv(dst_path + "Universities_test_mean.csv")

def median_imputation(df_train: pd.DataFrame, df_test: pd.DataFrame, dst_path: str) -> None:
    # ensure input data frame is not modified
    df_train_median_imputed = df_train.copy()
    df_test_median_imputed = df_test.copy()

    column_median_dict = impute_train_median(df_train_median_imputed, ["CWUR_score", "Student_satisfaction", "Founded_year"])
    column_mode_dict = impute_train_mode(df_train_median_imputed, ["Academic_Calender", "Campus_setting"]) # categorical columns must be imputed with mode

    # impute test data with the mean and mode of the training data, so that the test data is not leaked into the training data
    impute_test_median(df_test_median_imputed, column_median_dict)
    impute_test_mode(df_test_median_imputed, column_mode_dict)

    impute_constant(df_train_median_imputed, ["Academic_staff_from"], 5_000)
    impute_constant(df_test_median_imputed, ["Academic_staff_from"], 5_000)

    impute_constant(df_train_median_imputed, ["Academic_staff_to"], 10_000)
    impute_constant(df_test_median_imputed, ["Academic_staff_to"], 10_000)

    # drop university name, not needed anymore
    df_train_median_imputed.drop(columns="University_name", inplace=True)
    df_test_median_imputed.drop(columns="University_name", inplace=True)

    # save
    df_train_median_imputed.to_csv(dst_path + "Universities_train_median.csv")
    df_test_median_imputed.to_csv(dst_path + "Universities_test_median.csv")

def mixed_imputation(df_train: pd.DataFrame, df_test: pd.DataFrame, dst_path: str) -> None:
    df_train_mixed_imputed = df_train.copy()
    df_test_mixed_imputed = df_test.copy()

    impute_founded_year(df_train_mixed_imputed)
    # we assume that this information could be retrieved on the fly with new data samples during real-world application
    impute_founded_year(df_test_mixed_imputed)

    # in the following lines impute test data with statistics obtained from the training data, 
    # so that the test data is not leaked into the training data
    slope, intercept = impute_train_CWUR_score(df_train_mixed_imputed)
    impute_test_CWUR_score(df_test_mixed_imputed, slope, intercept)

    median = impute_train_student_satisfaction(df_train_mixed_imputed)
    impute_test_student_satisfaction(df_test_mixed_imputed, median)

    mode = impute_train_academic_calender(df_train_mixed_imputed)
    impute_test_academic_calender(df_test_mixed_imputed, mode)

    KNN_classifier = impute_train_campus_setting(df_train_mixed_imputed)
    impute_test_campus_setting(df_test_mixed_imputed, KNN_classifier)

    impute_constant(df_train_mixed_imputed, ["Academic_staff_from"], 5_000)
    impute_constant(df_test_mixed_imputed, ["Academic_staff_from"], 5_000)

    impute_constant(df_train_mixed_imputed, ["Academic_staff_to"], 10_000)
    impute_constant(df_test_mixed_imputed, ["Academic_staff_to"], 10_000)

    # drop university name, not needed anymore
    df_train_mixed_imputed.drop(columns="University_name", inplace=True)
    df_test_mixed_imputed.drop(columns="University_name", inplace=True)

    # save
    df_train_mixed_imputed.to_csv(dst_path + "Universities_train_mixed.csv")
    df_test_mixed_imputed.to_csv(dst_path + "Universities_test_mixed.csv")
    
if __name__ == "__main__":
    print("Imputing missing values ...")
    os.makedirs(DATA_PATH["imputation"], exist_ok=True)

    # load train set
    universities_train = pd.read_csv(DATA_PATH["split"] + "Universities_train.csv")
    universities_train.set_index("Id", inplace=True)

    # load test set
    universities_test = pd.read_csv(DATA_PATH["split"] + "Universities_test.csv")
    universities_test.set_index("Id", inplace=True)

    # impute with different methods
    mean_imputation(universities_train, universities_test, DATA_PATH["imputation"])
    median_imputation(universities_train, universities_test, DATA_PATH["imputation"])
    mixed_imputation(universities_train, universities_test, DATA_PATH["imputation"])
