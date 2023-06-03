import os
from pathlib import Path

import pandas as pd
import datetime as dt
import numpy as np

from src.utils import deduplication
os.chdir(Path(__file__).parents[2])

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Student_enrollment, Academic_staff, Control_type, 
# Academic_Calender, Campus_setting, Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website

print("Cleaning ...")
universities = pd.read_csv("data/Universities.csv")

universities.rename(columns={"Unnamed: 0": "Id"}, inplace=True)
universities.set_index("Id", inplace=True)

# correction of "9999" to "" in the founded year column
universities["Founded_year"].mask(universities["Founded_year"] > dt.datetime.now().year, None, inplace=True)

# conversion of percentage to float
universities["International_students"] = universities["International_students"].str.rstrip("%").astype("float") / 100
universities["Student_satisfaction"] = universities["Student_satisfaction"].str.rstrip("%").astype("float") / 100

# conversion of student enrollment to two columns
universities[["Student_enrollment_from", "Student_enrollment_to"]] = universities["Student_enrollment"].str.split("-", expand=True)
universities["Student_enrollment_from"] = universities["Student_enrollment_from"].str.replace(",", "").astype("float")
universities["Student_enrollment_to"] = universities["Student_enrollment_to"].str.replace(",", "").astype("float")
universities.drop(columns=["Student_enrollment"], inplace=True)

# conversion of academic staff to two columns
universities[["Academic_staff_from", "Academic_staff_to"]] = universities["Academic_staff"].str.split("-", expand=True)
universities.loc[universities["Academic_staff_from"] == "over", ["Academic_staff_to"]] = None
universities.loc[universities["Academic_staff_from"] == "over", ["Academic_staff_from"]] = "5000"
universities["Academic_staff_from"] = universities["Academic_staff_from"].str.replace(",", "").astype("float")
universities["Academic_staff_to"] = universities["Academic_staff_to"].str.replace(",", "").astype("float")
universities.drop(columns=["Academic_staff"], inplace=True)

# compare duplicates
universities_duplicates = universities.drop(universities.drop_duplicates(keep=False, inplace=False, ignore_index=False).index)
universities_duplicates.sort_values(by=["Motto"], inplace=True)

universities_duplicates_first = universities.drop(universities.drop_duplicates(keep="last", inplace=False, ignore_index=False).index)
universities_duplicates_first.sort_values(by=["Motto"], inplace=True)
universities_duplicates_first.set_index("Motto", inplace=True)

universities_duplicates_last = universities.drop(universities.drop_duplicates(keep="first", inplace=False, ignore_index=False).index)
universities_duplicates_last.sort_values(by=["Motto"], inplace=True)
universities_duplicates_last.set_index("Motto", inplace=True)

#print("comparison of duplicates: \n", universities_duplicates_first.compare(universities_duplicates_last))

# store
universities_duplicates.to_csv("data/Universities_cleaned_removed_duplicates.csv")
universities_duplicates_first.to_csv("data/Universities_cleaned_deduplicated_first.csv")
universities_duplicates_last.to_csv("data/Universities_cleaned_deduplicated_last.csv")

# drop duplicates
universities.drop_duplicates(keep="first", inplace=True)

# store
universities.to_csv("data/Universities_cleaned_deduplicated.csv")

universities = pd.read_csv("data/Universities.csv")

universities_deduplicated = universities.drop(columns='Unnamed: 0')
deduplication(universities_deduplicated)

#remove non valuable values with NaN
universities_deduplicated['Founded_year'] = universities_deduplicated['Founded_year'].replace(9999, np.NaN)
universities_deduplicated['Student_satisfaction'] = universities_deduplicated['Student_satisfaction'].replace(0.0, np.NaN)

#remove useless columns
useless_columns = np.array(['University_name', 'Motto', 'Website'])
universities_deduplicated = universities_deduplicated.drop(columns=useless_columns)

#% string into float
for column in universities_deduplicated.columns:
    if (universities_deduplicated[column].dtype == object):
        if (universities_deduplicated[column].str.contains('%').any()):
            universities_deduplicated[column] = universities_deduplicated[column].str.rstrip('%').astype('float') / 100.0

#save file
universities_deduplicated.to_csv("data/Universities_cleaned_deduplicated_new.csv")

