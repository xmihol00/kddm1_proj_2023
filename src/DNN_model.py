import pandas as pd
import tensorflow as tf
from tensorflow import keras as nn

train_mean = pd.read_csv("data/Universities_train_mean.csv")
train_median = pd.read_csv("data/Universities_train_median.csv")

test_mean = pd.read_csv("data/Universities_test_mean.csv")
test_median = pd.read_csv("data/Universities_test_median.csv")

# separate data frame to X and y
y_columns = ["UG_average_fees_(in_pounds)", "PG_average_fees_(in_pounds)"]
X_train_mean = train_mean.drop(columns=y_columns).to_numpy()
y_train_mean = train_mean[y_columns].to_numpy()

X_train_median = train_median.drop(columns=y_columns).to_numpy()
y_train_median = train_median[y_columns].to_numpy()

X_test_mean = test_mean.drop(columns=y_columns).to_numpy()
y_test_mean = test_mean[y_columns].to_numpy()

X_test_median = test_median.drop(columns=y_columns).to_numpy()
y_test_median = test_median[y_columns].to_numpy()

model = nn.Sequential([
    nn.layers.Dense(64, activation='relu', input_shape=[X_train_mean.shape[1]]),
    nn.layers.Dense(64, activation='relu'),
    nn.layers.Dense(2)
])

# train the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
model.fit(X_train_mean, y_train_mean, epochs=50, verbose=1)

# evaluate the model on the test set
results = model.evaluate(X_test_mean, y_test_mean, verbose=2)
print(results)
