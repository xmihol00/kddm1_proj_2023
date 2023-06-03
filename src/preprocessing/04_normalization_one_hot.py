import os
from pathlib import Path
import pandas as pd
import numpy as np

from src.seed import RANDOM_SEED
from sklearn.preprocessing import StandardScaler, OneHotEncoder

os.chdir(Path(__file__).parents[2])

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to

print("Normaling continuous features and one-hot encoding categorical features ...")

# zero mean and unit variance normalization for continuous features (apart from the target variables)
# normalized columns (continuous features):
normalized_continuous_columns = ["UK_rank", 
                                 "World_rank", 
                                 "CWUR_score", 
                                 "Minimum_IELTS_score", 
                                 "International_students", 
                                 "Student_satisfaction", 
                                 "Estimated_cost_of_living_per_year_(in_pounds)", 
                                 "Student_enrollment_from", 
                                 "Student_enrollment_to", 
                                 "Academic_staff_from", 
                                 "Academic_staff_to"]

# one-hot encoding for categorical features
# one-hot encoded columns (categorical features):
one_hot_encoded_columns = ["Region",
                           "Control_type",
                           "Academic_Calender",
                           "Campus_setting"]


# make sure test data don't leak to the training data, i.e. mean and variance are calculated only for the training data

universities_train_mean_imputed = pd.read_csv("data/Universities_train_mean_imputed.csv")
universities_train_mean_imputed.set_index("Id", inplace=True)
universities_train_median_imputed = pd.read_csv("data/Universities_train_median_imputed.csv")
universities_train_median_imputed.set_index("Id", inplace=True)

universities_test_mean_imputed = pd.read_csv("data/Universities_test_mean_imputed.csv")
universities_test_mean_imputed.set_index("Id", inplace=True)
universities_test_median_imputed = pd.read_csv("data/Universities_test_median_imputed.csv")
universities_test_median_imputed.set_index("Id", inplace=True)

# deep copy
universities_train_mean_imputed_normalized = universities_train_mean_imputed.copy()
universities_train_median_imputed_normalized = universities_train_median_imputed.copy()

universities_test_mean_imputed_normalized = universities_test_mean_imputed.copy()
universities_test_median_imputed_normalized = universities_test_median_imputed.copy()

# normalization for continuous features
mean = universities_train_mean_imputed[normalized_continuous_columns].mean()
std = universities_train_mean_imputed[normalized_continuous_columns].std()

universities_train_mean_imputed_normalized[normalized_continuous_columns] = (universities_train_mean_imputed[normalized_continuous_columns] - mean) / std
universities_train_median_imputed_normalized[normalized_continuous_columns] = (universities_train_median_imputed[normalized_continuous_columns] - mean) / std

universities_test_mean_imputed_normalized[normalized_continuous_columns] = (universities_test_mean_imputed[normalized_continuous_columns] - mean) / std
universities_test_median_imputed_normalized[normalized_continuous_columns] = (universities_test_median_imputed[normalized_continuous_columns] - mean) / std

# one-hot encoding for categorical features
universities_train_mean_imputed_normalized = pd.get_dummies(universities_train_mean_imputed_normalized, columns=one_hot_encoded_columns)
universities_train_median_imputed_normalized = pd.get_dummies(universities_train_median_imputed_normalized, columns=one_hot_encoded_columns)

universities_test_mean_imputed_normalized = pd.get_dummies(universities_test_mean_imputed_normalized, columns=one_hot_encoded_columns)
universities_test_median_imputed_normalized = pd.get_dummies(universities_test_median_imputed_normalized, columns=one_hot_encoded_columns)

# add columns that might be missing in the test data due to one-hot encoding
missing_test_columns = set(universities_train_mean_imputed_normalized.columns) - set(universities_test_mean_imputed_normalized.columns)
for missing_test_column in missing_test_columns:
    universities_test_mean_imputed_normalized[missing_test_column] = 0
    universities_test_median_imputed_normalized[missing_test_column] = 0

# add columns that might be missing in the training data due to one-hot encoding
missing_train_columns = set(universities_test_mean_imputed_normalized.columns) - set(universities_train_mean_imputed_normalized.columns)
for missing_train_column in missing_train_columns:
    universities_train_mean_imputed_normalized[missing_train_column] = 0
    universities_train_median_imputed_normalized[missing_train_column] = 0

# order columns alphabetically for both train and test data
universities_train_mean_imputed_normalized = universities_train_mean_imputed_normalized.reindex(sorted(universities_train_mean_imputed_normalized.columns), axis=1)
universities_train_median_imputed_normalized = universities_train_median_imputed_normalized.reindex(sorted(universities_train_median_imputed_normalized.columns), axis=1)

universities_test_mean_imputed_normalized = universities_test_mean_imputed_normalized.reindex(sorted(universities_test_mean_imputed_normalized.columns), axis=1)
universities_test_median_imputed_normalized = universities_test_median_imputed_normalized.reindex(sorted(universities_test_median_imputed_normalized.columns), axis=1)

# save to csv
universities_train_mean_imputed_normalized.to_csv("data/Universities_train_mean_imputed_normalized.csv")
universities_train_median_imputed_normalized.to_csv("data/Universities_train_median_imputed_normalized.csv")

universities_test_mean_imputed_normalized.to_csv("data/Universities_test_mean_imputed_normalized.csv")
universities_test_median_imputed_normalized.to_csv("data/Universities_test_median_imputed_normalized.csv")


universities_mixed_imputed = pd.read_csv("data/Universities_mixed_imputed.csv")
universities_mixed_imputed = universities_mixed_imputed.drop(columns='Unnamed: 0')

universities_mixed_imputed_normalized = pd.DataFrame()
for column in universities_mixed_imputed.columns:
    if((universities_mixed_imputed[column].dtype == 'int') | (universities_mixed_imputed[column].dtype == 'float')):
        scaler = StandardScaler()
        df_num = pd.DataFrame(scaler.fit_transform(universities_mixed_imputed[[column]]))
        df_num.rename(columns={0 : column}, inplace=True)
        universities_mixed_imputed_normalized = pd.concat([universities_mixed_imputed_normalized, df_num], axis=1)
    else:
        encoder = OneHotEncoder()
        df_cat = pd.DataFrame(encoder.fit_transform(universities_mixed_imputed[[column]]).toarray())
        column_names = np.array(encoder.categories_)
        df_cat = df_cat.set_axis(column_names.flatten(), axis=1, copy=False)
        universities_mixed_imputed_normalized = pd.concat([universities_mixed_imputed_normalized, df_cat], axis=1)

universities_mixed_imputed_normalized.to_csv("data/Universities_mixed_imputed_normalized.csv", index=False)

universities_train_mixed_imputed_normalized = universities_mixed_imputed_normalized.sample(frac=0.8, random_state=RANDOM_SEED)
universities_test_mixed_imputed_normalized = universities_mixed_imputed_normalized.drop(universities_train_mixed_imputed_normalized.index)

universities_train_mixed_imputed_normalized.to_csv("data/Universities_train_mixed_imputed_normalized.csv", index=False)
universities_test_mixed_imputed_normalized.to_csv("data/Universities_test_mixed_imputed_normalized.csv", index=False)


