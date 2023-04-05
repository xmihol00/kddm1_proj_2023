import pandas as pd
import numpy as np
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
    nn.layers.Dense(X_train_mean.shape[1], activation='relu', input_shape=[X_train_mean.shape[1]]),
    nn.layers.Dense(X_train_mean.shape[1], activation='relu'),
    nn.layers.Dense(X_train_mean.shape[1], activation='linear'),
    nn.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
# train the model
model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train_mean, y_train_mean[:, 0], epochs=100, verbose=2)

# evaluate the model on the test set
result = model.predict(X_test_mean)
print(np.vstack((result.reshape(result.shape[0]), y_test_mean[:, 0])).T)
