import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import DATA_PATH

train_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mean.csv")
train_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_median.csv")
train_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_train_mixed.csv")

test_mean = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mean.csv")
test_median = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_median.csv")
test_mixed = pd.read_csv(DATA_PATH["numeric"] + "Universities_test_mixed.csv")

# separate data frame to X and y
target_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
X_train_mean = train_mean.drop(columns=target_columns).to_numpy()
y_train_mean = train_mean[target_columns].to_numpy()
X_train_median = train_median.drop(columns=target_columns).to_numpy()
y_train_median = train_median[target_columns].to_numpy()
X_train_mixed = train_mixed.drop(columns=target_columns).to_numpy()
y_train_mixed = train_mixed[target_columns].to_numpy()

X_test_mean = test_mean.drop(columns=target_columns).to_numpy()
y_test_mean = test_mean[target_columns].to_numpy()
X_test_median = test_median.drop(columns=target_columns).to_numpy()
y_test_median = test_median[target_columns].to_numpy()
X_test_mixed = test_mixed.drop(columns=target_columns).to_numpy()
y_test_mixed = test_mixed[target_columns].to_numpy()

# train on mean imputed data
model_mean = LinearRegression()
model_mean.fit(X_train_mean, y_train_median)

# predict
y_pred = model_mean.predict(X_test_mean)

# evaluate
mse = metrics.mean_squared_error(y_test_mean, y_pred)
rmse = metrics.mean_squared_error(y_test_mean, y_pred, squared=False)
mae = metrics.mean_absolute_error(y_test_mean, y_pred)
r2 = metrics.r2_score(y_test_mean, y_pred)

print("Test results:")
print(f"  - baseline linear model trained on all columns with mean value imputation MSE: {mse}")
print(f"  - baseline linear model trained on all columns with mean value imputation RMSE: {rmse}")
print(f"  - baseline linear model trained on all columns with mean value imputation MAE: {mae}")
print(f"  - baseline linear model trained on all columns with mean value imputation R2 Score: {r2}", end="\n\n")

# train on median imputed data
model_median = LinearRegression()
model_median.fit(X_train_median, y_train_median)

# predict
y_pred = model_median.predict(X_test_median)

# evaluate
mse = metrics.mean_squared_error(y_test_median, y_pred)
rmse = metrics.mean_squared_error(y_test_median, y_pred, squared=False)
mae = metrics.mean_absolute_error(y_test_median, y_pred)
r2 = metrics.r2_score(y_test_median, y_pred)

print(f"  - baseline linear model trained on all columns with median value imputation MSE: {mse}")
print(f"  - baseline linear model trained on all columns with median value imputation RMSE: {rmse}")
print(f"  - baseline linear model trained on all columns with median value imputation MAE: {mae}")
print(f"  - baseline linear model trained on all columns with median value imputation R2 Score: {r2}", end="\n\n")

# train on mixed imputed data
model_mixed = LinearRegression()
model_mixed.fit(X_train_mixed, y_train_mixed)

# predict
y_pred = model_mixed.predict(X_test_mixed)

# evaluate
mse = metrics.mean_squared_error(y_test_mixed, y_pred)
rmse = metrics.mean_squared_error(y_test_mixed, y_pred, squared=False)
mae = metrics.mean_absolute_error(y_test_mixed, y_pred)
r2 = metrics.r2_score(y_test_mixed, y_pred)

print(f"  - baseline linear model trained on all columns with mixed value imputation MSE: {mse}")
print(f"  - baseline linear model trained on all columns with mixed value imputation RMSE: {rmse}")
print(f"  - baseline linear model trained on all columns with mixed value imputation MAE: {mae}")
print(f"  - baseline linear model trained on all columns with mixed value imputation R2 Score: {r2}", end="\n\n")

# store predicted and ground truth values of the best imputed data
predicted_truth = pd.DataFrame({"predicted UG_average_fees_(in_pounds)": y_pred[:, 0], "predicted PG_average_fees_(in_pounds)": y_pred[:, 1], 
                                "truth UG_average_fees_(in_pounds)": y_test_median[:, 0], "truth PG_average_fees_(in_pounds)": y_test_median[:, 1]})

print("Predicted vs. ground truth values mixed imputed:")
print(predicted_truth)
predicted_truth.to_csv("results/baseline_predicted_truth.csv", index=False)

# print results table
results = pd.DataFrame({"MSE": [mse], "RMSE": [rmse], "MAE": [mae], "R2 Score": [r2]})
print()
print(results.to_markdown())
