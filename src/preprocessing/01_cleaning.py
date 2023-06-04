import os, sys
from pathlib import Path

import pandas as pd
import datetime as dt
import numpy as np

os.chdir(Path(__file__).parents[2])
sys.path.append(os.getcwd())

from src.utils import deduplication


################################################################################
def cleaning(df: pd.DataFrame):    
    df.rename(columns={"Unnamed: 0": "Id"}, inplace=True)
    df.set_index("Id", inplace=True)
    
    # correction of "9999" to "" in the founded year column
    df["Founded_year"].mask(df["Founded_year"] > dt.datetime.now().year, None, inplace=True)

    # conversion of percentage to float
    df["International_students"] = df["International_students"].str.rstrip("%").astype("float") / 100
    df["Student_satisfaction"] = df["Student_satisfaction"].str.rstrip("%").astype("float") / 100

    # conversion of student enrollment to two columns
    df[["Student_enrollment_from", "Student_enrollment_to"]] = df["Student_enrollment"].str.split("-", expand=True)
    df["Student_enrollment_from"] = df["Student_enrollment_from"].str.replace(",", "").astype("float")
    df["Student_enrollment_to"] = df["Student_enrollment_to"].str.replace(",", "").astype("float")
    df.drop(columns=["Student_enrollment"], inplace=True)

    # conversion of academic staff to two columns
    df[["Academic_staff_from", "Academic_staff_to"]] = df["Academic_staff"].str.split("-", expand=True)
    df.loc[df["Academic_staff_from"] == "over", ["Academic_staff_to"]] = None
    df.loc[df["Academic_staff_from"] == "over", ["Academic_staff_from"]] = "5000"
    df["Academic_staff_from"] = df["Academic_staff_from"].str.replace(",", "").astype("float")
    df["Academic_staff_to"] = df["Academic_staff_to"].str.replace(",", "").astype("float")
    df.drop(columns=["Academic_staff"], inplace=True)

    # drop duplicates
    df.drop_duplicates(keep="first", inplace=True)

def cleaning_by_thomas(df: pd.DataFrame):
    df_ret = universities.drop(columns='Unnamed: 0')
    deduplication(df_ret)
    
    #remove non valuable values with NaN
    df_ret['Founded_year'] = df_ret['Founded_year'].replace(9999, np.NaN)
    df_ret['Student_satisfaction'] = df_ret['Student_satisfaction'].replace(0.0, np.NaN)

    #remove useless columns
    useless_columns = np.array(['University_name', 'Motto', 'Website'])
    df_ret = df_ret.drop(columns=useless_columns)

    #% string into float
    for column in df_ret.columns:
        if (df_ret[column].dtype == object):
            if (df_ret[column].str.contains('%').any()):
                df_ret[column] = df_ret[column].str.rstrip('%').astype('float') / 100.0

    return df_ret


################################################################################
print("Cleaning ...")

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Student_enrollment, Academic_staff, Control_type, 
# Academic_Calender, Campus_setting, Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website


################################################################################
# cleaning
universities = pd.read_csv("data/Universities.csv")
cleaning(universities)

# store
universities.to_csv("data/Universities_cleaned_deduplicated.csv")


################################################################################
# cleaning by thomas
universities = pd.read_csv("data/Universities.csv")
universities_deduplicated = cleaning_by_thomas(universities)

#save file
universities_deduplicated.to_csv("data/Universities_cleaned_deduplicated_by_thomas.csv")


################################################################################
# compare duplicates
universities_duplicates       = universities.drop(universities.drop_duplicates(keep=False  , inplace=False, ignore_index=False).index)
universities_duplicates_first = universities.drop(universities.drop_duplicates(keep="last" , inplace=False, ignore_index=False).index)
universities_duplicates_last  = universities.drop(universities.drop_duplicates(keep="first", inplace=False, ignore_index=False).index)

sort_by_col = "University_name"
universities_duplicates.sort_values(by=[sort_by_col], inplace=True)
universities_duplicates_first.sort_values(by=[sort_by_col], inplace=True)
universities_duplicates_last.sort_values(by=[sort_by_col], inplace=True)
# universities_duplicates_first.set_index(sort_by_col, inplace=True)
# universities_duplicates_last.set_index(sort_by_col, inplace=True)

# print("comparison of duplicates: \n", universities_duplicates_first.compare(universities_duplicates_last))

# store
universities_duplicates.to_csv("data/Universities_cleaned_removed_duplicates.csv")
universities_duplicates_first.to_csv("data/Universities_cleaned_removed_duplicates_first.csv")
universities_duplicates_last.to_csv("data/Universities_cleaned_removed_duplicates_last.csv")