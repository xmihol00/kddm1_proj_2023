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

def split(df : pd.DataFrame, test_size : float):
    np.random.seed(RANDOM_SEED)
    size = len(df.index)
    test = test_size
    test_items = round(size * test)
    test_indices = np.random.choice(df.dropna().index, test_items, replace=False)
    train_indices = [i for i in range(size) if i not in test_indices]
    test_indices.sort()
    df_test = df.iloc[test_indices]
    df_train = df.iloc[train_indices]
    return df_train, df_test

def imputation_numeric(df: pd.DataFrame):
    lm = LinearRegression()
    df_regression = df[df['CWUR_score'].notna()]
    lm.fit(df_regression['UG_average_fees_(in_pounds)'].to_numpy().reshape(-1, 1), df_regression['CWUR_score'])
    slope = lm.coef_
    intercept = lm.intercept_

    df_numeric = df.select_dtypes(include=["float", 'int'])
    for index, value in enumerate(df_numeric['UG_average_fees_(in_pounds)']):
        if(pd.isna(df.loc[index, 'CWUR_score'])):
            #switch between median or linear regression imputation
            df.loc[index, 'CWUR_score'] = (df.loc[index, 'UG_average_fees_(in_pounds)'] * slope + intercept)[0]
            #df.loc[index, 'CWUR_score'] = df['CWUR_score'].median()
        if(pd.isna(df_numeric.loc[index, 'Student_satisfaction'])):
            df.loc[index, 'Student_satisfaction'] = df['Student_satisfaction'].median()


def imputation_founded_year(df : pd.DataFrame):
    nan_indices = np.where(df['Founded_year'].isna())[0]        # returns tuple(x)
    founded_years = [1096, 1413, 1964, 1965, 1966, 1904, 1900, 2012, 1952, 1963, 1829, 1920,
                     1964, 1961, 1872, 1895, 2006, 2005, 1992, 1850, 2005, 1994, 1966, 2007,
                     1968, 2005, 2007, 2006]
    df.loc[nan_indices, 'Founded_year'] = founded_years
    return df

def imputation_academic_calender(df: pd.DataFrame):
    mode = df['Academic_Calender'].value_counts().idxmax()
    df['Academic_Calender'] = df['Academic_Calender'].fillna(mode)

def imputation_campus_setting(df : pd.DataFrame):
    df_kmeans = df[['Latitude', 'Longitude', 'Campus_setting']].copy()

    test, train = split(df_kmeans, test_size=0.7)
    train_X = train.drop(columns='Campus_setting')
    train_y = train['Campus_setting']

    kneighbors = KNeighborsClassifier(n_neighbors=4, p=2)
    kneighbors.fit(train_X, train_y)

    predict_indices = np.where(df['Campus_setting'].isna())
    df_predict = test.drop(columns='Campus_setting').loc[predict_indices]
    predicted_values = kneighbors.predict(df_predict).tolist()
    for index in predict_indices:
        df.loc[index, 'Campus_setting'] = predicted_values.pop(0)
