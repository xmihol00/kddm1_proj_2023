import pandas as pd
import numpy as np

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Student_enrollment, Academic_staff, Control_type, 
# Academic_Calender, Campus_setting, Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website

df = pd.read_csv("data/Universities.csv")

################################################################################
# Non-Null Count, Dtype
print('\n' + '#'*120)
df.info()

################################################################################
# Null Count per column
print('\n' + '#'*120)

nan_col_count = df.isna().sum()
for i, x in enumerate(nan_col_count):
  print("{:3} {:50} {:3}".format(i, df.columns[i], x))
print("sum over columns:     {:}".format(nan_col_count.sum()))
print("nr columns with null: {:}".format(np.sum(nan_col_count != 0)))

################################################################################
# Null Count per row
print('\n' + '#'*120)

nan_row_count = df.isna().sum(axis=1)
unique, counts = np.unique(nan_row_count, return_counts=True)
print("nr of null per row:    ", unique)
print("count:                 ", counts)
print("nr rows contain null:  ", np.sum(nan_row_count != 0))
print("nr rows are compleat:  ", np.sum(nan_row_count == 0))

# ################################################################################
# # continuous columns
# print(df.select_dtypes(include='number'))
# # categorical columns
# print(df.select_dtypes(exclude='number'))

###############################################################################
print('\n' + '#'*120)
print(df.describe())

