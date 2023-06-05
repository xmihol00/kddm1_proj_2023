import os
import sys
import pandas as pd
import datetime as dt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # ensure includes of our files work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import DATA_PATH

# data set header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Student_enrollment, Academic_staff, Control_type, 
# Academic_Calender, Campus_setting, Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website

def clean_and_deduplicate(df: pd.DataFrame):
    # set the first column as index    
    df.rename(columns={"Unnamed: 0": "Id"}, inplace=True)
    df.set_index("Id", inplace=True)

    # deduplication
    duplicated_rows = df.index[df['Latitude'].duplicated()].to_list()
    df.drop(duplicated_rows, axis=0, inplace=True)
    
    # correction of dates in the future, i.e. "9999", to NaN in the founded year column
    df["Founded_year"].mask(df["Founded_year"] > dt.datetime.now().year, np.NaN, inplace=True)

    # conversion of percentage to float
    df["International_students"] = df["International_students"].str.rstrip("%").astype("float") / 100
    df["Student_satisfaction"] = df["Student_satisfaction"].str.rstrip("%").astype("float") / 100
    df['Student_satisfaction'] = df['Student_satisfaction'].replace(0.0, np.NaN)

    # conversion of student enrollment to two columns
    df[["Student_enrollment_from", "Student_enrollment_to"]] = df["Student_enrollment"].str.split("-", expand=True)
    df["Student_enrollment_from"] = df["Student_enrollment_from"].str.replace(",", "").astype("float")
    df["Student_enrollment_to"] = df["Student_enrollment_to"].str.replace(",", "").astype("float")
    df.drop(columns=["Student_enrollment"], inplace=True)

    # conversion of academic staff to two columns
    df[["Academic_staff_from", "Academic_staff_to"]] = df["Academic_staff"].str.split("-", expand=True)
    df.loc[df["Academic_staff_from"] == "over", ["Academic_staff_to"]] = np.NaN
    df.loc[df["Academic_staff_from"] == "over", ["Academic_staff_from"]] = "5000"
    df["Academic_staff_from"] = df["Academic_staff_from"].str.replace(",", "").astype("float")
    df["Academic_staff_to"] = df["Academic_staff_to"].str.replace(",", "").astype("float")
    df.drop(columns=["Academic_staff"], inplace=True)

    # removal of unused columns
    useless_columns = np.array(['University_name', 'Motto', 'Website'])
    df = df.drop(columns=useless_columns)

if __name__ == "__main__":
    print("Cleaning ...")

    os.makedirs(DATA_PATH["original"], exist_ok=True)    # make sure the output directory exists
    
    universities = pd.read_csv(DATA_PATH["original"] + "Universities.csv")
    clean_and_deduplicate(universities)

    universities.to_csv(DATA_PATH["cleaning"] + "Universities_cleaned_deduplicated.csv")
