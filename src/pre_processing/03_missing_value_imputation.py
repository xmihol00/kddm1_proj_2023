import pandas as pd

# header:
# Id, University_name, Region, Founded_year, Motto, UK_rank, World_rank, CWUR_score, Minimum_IELTS_score, UG_average_fees_(in_pounds), 
# PG_average_fees_(in_pounds), International_students, Student_satisfaction, Control_type, Academic_Calender, Campus_setting, 
# Estimated_cost_of_living_per_year_(in_pounds), Latitude, Longitude, Website, Student_enrollment_from, Student_enrollment_to,
# Academic_staff_from, Academic_staff_to

# make sure test data don't leak to the training data, i.e. mean, median, mode are calculated separately

universities_train = pd.read_csv("data/Universities_train_split.csv")
universities_train.set_index("Id", inplace=True)

universities_test = pd.read_csv("data/Universities_test_split.csv")
universities_test.set_index("Id", inplace=True)

universities_train_mean_imputed = universities_train.copy()
universities_train_median_imputed = universities_train.copy()
universities_test_mean_imputed = universities_test.copy()
universities_test_median_imputed = universities_test.copy()

# imputation of the CWUR score
CWUR_score_mean = universities_train["CWUR_score"].mean()
CWUR_score_median = universities_train["CWUR_score"].median()
universities_train_mean_imputed["CWUR_score"].fillna(CWUR_score_mean, inplace=True)
universities_train_median_imputed["CWUR_score"].fillna(CWUR_score_median, inplace=True)

CWUR_score_mean = universities_test["CWUR_score"].mean()
CWUR_score_median = universities_test["CWUR_score"].median()
universities_test_mean_imputed["CWUR_score"].fillna(CWUR_score_mean, inplace=True)
universities_test_median_imputed["CWUR_score"].fillna(CWUR_score_median, inplace=True)

# imputation of the student satisfaction
student_satisfaction_mean = universities_train["Student_satisfaction"].mean()
student_satisfaction_median = universities_train["Student_satisfaction"].median()
universities_train_mean_imputed["Student_satisfaction"].fillna(student_satisfaction_mean, inplace=True)
universities_train_median_imputed["Student_satisfaction"].fillna(student_satisfaction_median, inplace=True)

student_satisfaction_mean = universities_test["Student_satisfaction"].mean()
student_satisfaction_median = universities_test["Student_satisfaction"].median()
universities_test_mean_imputed["Student_satisfaction"].fillna(student_satisfaction_mean, inplace=True)
universities_test_median_imputed["Student_satisfaction"].fillna(student_satisfaction_median, inplace=True)

# imputation of the academic calendar (categorical)
academic_calendar_mode = universities_train["Academic_Calender"].mode()[0]
universities_train_mean_imputed["Academic_Calender"].fillna(academic_calendar_mode, inplace=True)
universities_train_median_imputed["Academic_Calender"].fillna(academic_calendar_mode, inplace=True)

academic_calendar_mode = universities_test["Academic_Calender"].mode()[0]
universities_test_mean_imputed["Academic_Calender"].fillna(academic_calendar_mode, inplace=True)
universities_test_median_imputed["Academic_Calender"].fillna(academic_calendar_mode, inplace=True)

# imputation of the campus setting (categorical)
campus_setting_mode = universities_train["Campus_setting"].mode()[0]
universities_train_mean_imputed["Campus_setting"].fillna(campus_setting_mode, inplace=True)
universities_train_median_imputed["Campus_setting"].fillna(campus_setting_mode, inplace=True)

campus_setting_mode = universities_test["Campus_setting"].mode()[0]
universities_test_mean_imputed["Campus_setting"].fillna(campus_setting_mode, inplace=True)
universities_test_median_imputed["Campus_setting"].fillna(campus_setting_mode, inplace=True)

# imputation of the academic staff to 10_000
universities_train_mean_imputed["Academic_staff_to"].fillna(10_000, inplace=True)
universities_train_median_imputed["Academic_staff_to"].fillna(10_000, inplace=True)
universities_test_mean_imputed["Academic_staff_to"].fillna(10_000, inplace=True)
universities_test_median_imputed["Academic_staff_to"].fillna(10_000, inplace=True)

# save the imputed datasets
universities_train_mean_imputed.to_csv("data/Universities_train_mean_imputed.csv")
universities_train_median_imputed.to_csv("data/Universities_train_median_imputed.csv")

universities_test_mean_imputed.to_csv("data/Universities_test_mean_imputed.csv")
universities_test_median_imputed.to_csv("data/Universities_test_median_imputed.csv")
