import pandas as pd
import numpy as np

from src.seed import RANDOM_SEED
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


def deduplication(df : pd.DataFrame):
    duplicated_rows = df.index[df['Latitude'].duplicated()].to_list()
    df.drop(duplicated_rows, axis=0, inplace=True)
    return df

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
    nan_indices = np.where(df['Founded_year'].isna())
    founded_years = [1096, 1413, 1964, 1965, 1966, 1904, 1900, 1901, 1862, 1963, 1829, 1920,
                     1964, 1961, 1872, 1895, 1882, 1844, 1870, 1850, 1840, 1994, 1832, 1891,
                     1968, 2005, 1822, 2006]
    for index in nan_indices:
        df.loc[index, 'Founded_year'] = founded_years.pop(0)

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
