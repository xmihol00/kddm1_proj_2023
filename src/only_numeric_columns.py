import pandas as pd

universities_train_mean = pd.read_csv("data/Universities_train_mean_imputed_normalized.csv")
universities_train_median = pd.read_csv("data/Universities_train_median_imputed_normalized.csv")

universities_test_mean = pd.read_csv("data/Universities_test_mean_imputed_normalized.csv")
universities_test_median = pd.read_csv("data/Universities_test_median_imputed_normalized.csv")

# drop non numeric columns, columns with NaNs and the Id column
universities_train_mean = universities_train_mean.select_dtypes(include=['float64', 'int64'])
universities_train_median = universities_train_median.select_dtypes(include=['float64', 'int64'])
universities_train_mean.dropna(axis=1, inplace=True)
universities_train_median.dropna(axis=1, inplace=True)
universities_train_mean.drop(columns=["Id"], inplace=True)
universities_train_median.drop(columns=["Id"], inplace=True)

universities_test_mean = universities_test_mean.select_dtypes(include=['float64', 'int64'])
universities_test_median = universities_test_median.select_dtypes(include=['float64', 'int64'])
universities_test_mean.dropna(axis=1, inplace=True)
universities_test_median.dropna(axis=1, inplace=True)
universities_test_mean.drop(columns=["Id"], inplace=True)
universities_test_median.drop(columns=["Id"], inplace=True)

# save the data to final train and test files
universities_train_mean.to_csv("data/Universities_train_mean.csv", index=False)
universities_train_median.to_csv("data/Universities_train_median.csv", index=False)

universities_test_mean.to_csv("data/Universities_test_mean.csv", index=False)
universities_test_median.to_csv("data/Universities_test_median.csv", index=False)
