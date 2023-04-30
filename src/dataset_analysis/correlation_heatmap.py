import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pointbiserialr, chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denominator = min((kcorr-1), (rcorr-1))
    if denominator <= 0:
        return 0
    return np.sqrt(phi2corr / denominator)

os.makedirs("plots", exist_ok=True)

data_df = pd.read_csv("data/Universities_cleaned_deduplicated.csv")

continuous_cols = ["UK_rank", 
                   "World_rank", 
                   "CWUR_score", 
                   "Minimum_IELTS_score", 
                   "UG_average_fees_(in_pounds)",
                   "PG_average_fees_(in_pounds)",
                   "International_students", 
                   "Student_satisfaction", 
                   "Estimated_cost_of_living_per_year_(in_pounds)", 
                   "Student_enrollment_from", 
                   "Student_enrollment_to", 
                   "Academic_staff_from", 
                   "Academic_staff_to"]

categorical_cols = ["Control_type",
                    "Academic_Calender",
                    "Campus_setting"]

# select columns for correlation
selected_columns = continuous_cols + categorical_cols

correlation_matrix_df = pd.DataFrame(index=selected_columns, columns=selected_columns)
data_df = data_df[selected_columns]
data_df[categorical_cols] = data_df[categorical_cols].astype("category")

for col1 in selected_columns:
    for col2 in selected_columns:
        if col1 in continuous_cols and col2 in continuous_cols:
            # Pearson's correlation for continuous-continuous pairs
            if data_df[col1].nunique() == 1 or data_df[col2].nunique() == 1:
                correlation_matrix_df.loc[col1, col2] = np.nan
            else:
                correlation_matrix_df.loc[col1, col2] = data_df[col1].corr(data_df[col2])
        elif col1 in categorical_cols and col2 in categorical_cols:
            # Cramér's V for categorical-categorical pairs
            correlation_matrix_df.loc[col1, col2] = cramers_v(data_df[col1], data_df[col2])
        else:
            # Point-biserial correlation for continuous-categorical pairs
            continuous_var, categorical_var = (col1, col2) if col1 in continuous_cols else (col2, col1)
            if data_df[continuous_var].nunique() == 1 or data_df[categorical_var].nunique() == 1:
                correlation_matrix_df.loc[col1, col2] = np.nan
            else:
                correlation_matrix_df.loc[col1, col2] = pointbiserialr(data_df[categorical_var].cat.codes, data_df[continuous_var].fillna(data_df[continuous_var].mean()))[0]

correlation_matrix_df = correlation_matrix_df.astype(float)
correlation_matrix_df = correlation_matrix_df.round(2)

# create correlation heatmap
plt.figure(figsize=(17, 11))
sns.heatmap(correlation_matrix_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 12})
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=500)
plt.show()
