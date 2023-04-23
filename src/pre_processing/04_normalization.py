import pandas as pd

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to

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

