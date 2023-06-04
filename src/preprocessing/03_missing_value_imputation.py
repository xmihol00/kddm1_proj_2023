import os
import pandas as pd
from pathlib import Path

from src.utils import imputation_numeric
from src.utils import imputation_founded_year
from src.utils import imputation_academic_calender
from src.utils import imputation_campus_setting
from src.seed import RANDOM_SEED
################################################################################
def imputation_mean(df_dst: pd.DataFrame, df_src: pd.DataFrame, columns: str):
    for col in columns: 
        mean = df_src[col].mean()
        df_dst[col].fillna(mean, inplace=True)

def imputation_median(df_dst: pd.DataFrame, df_src: pd.DataFrame, columns: str):
    for col in columns: 
        median = df_src[col].median()
        df_dst[col].fillna(median, inplace=True)

def imputation_mode(df_dst: pd.DataFrame, df_src: pd.DataFrame, columns: str):
    for col in columns: 
        median = df_src[col].mode()
        df_dst[col].fillna(median, inplace=True)
    
################################################################################
os.chdir(Path(__file__).parents[2])
print("Imputing missing values ...")


################################################################################
# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to

# make sure test data don't leak to the training data, i.e. mean, median, mode are calculated separately

################################################################################
# read
universities_train = pd.read_csv("data/Universities_train_split.csv")
universities_train.set_index("Id", inplace=True)

universities_test = pd.read_csv("data/Universities_test_split.csv")
universities_test.set_index("Id", inplace=True)


################################################################################
# mean
universities_train_mean_imputed = universities_train.copy()
universities_test_mean_imputed = universities_test.copy()

imputation_mean(universities_train_mean_imputed, universities_train, ['CWUR_score', 'Student_satisfaction'])
imputation_mean(universities_test_mean_imputed , universities_test , ['CWUR_score', 'Student_satisfaction'])

imputation_mode(universities_train_mean_imputed, universities_train, ['Academic_Calender', 'Campus_setting'])
imputation_mode(universities_test_mean_imputed , universities_test , ['Academic_Calender', 'Campus_setting'])

# imputation of the academic staff to 10_000
universities_train_mean_imputed["Academic_staff_to"].fillna(10_000, inplace=True)
universities_test_mean_imputed["Academic_staff_to"].fillna(10_000, inplace=True)

# save
universities_train_mean_imputed.to_csv("data/Universities_train_mean_imputed.csv")
universities_test_mean_imputed.to_csv("data/Universities_test_mean_imputed.csv")


################################################################################
# median
universities_train_median_imputed = universities_train.copy()
universities_test_median_imputed = universities_test.copy()

imputation_median(universities_train_median_imputed, universities_train, ['CWUR_score', 'Student_satisfaction'])
imputation_median(universities_test_median_imputed , universities_test , ['CWUR_score', 'Student_satisfaction'])

imputation_mode(universities_train_median_imputed, universities_train, ['Academic_Calender', 'Campus_setting'])
imputation_mode(universities_test_median_imputed , universities_test , ['Academic_Calender', 'Campus_setting'])

# imputation of the academic staff to 10_000
universities_train_median_imputed["Academic_staff_to"].fillna(10_000, inplace=True)
universities_test_median_imputed["Academic_staff_to"].fillna(10_000, inplace=True)

# save
universities_train_median_imputed.to_csv("data/Universities_train_median_imputed.csv")
universities_test_median_imputed.to_csv("data/Universities_test_median_imputed.csv")


################################################################################
# by thomas
universities = pd.read_csv("data/Universities_cleaned_deduplicated_new.csv")
universities = universities.drop(columns='Unnamed: 0')

imputation_numeric(universities)
imputation_founded_year(universities)
imputation_academic_calender(universities)
imputation_campus_setting(universities)

universities_train = universities.sample(frac=0.8, random_state=RANDOM_SEED)
universities_test = universities.drop(universities_train.index)

universities.to_csv("data/Universities_mixed_imputed.csv")
universities_train.to_csv("data/Universities_train_mixed_imputed.csv")
universities_test.to_csv("data/Universities_test_mixed_imputed.csv")

