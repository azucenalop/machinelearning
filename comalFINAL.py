
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
os.chdir('C:/Users/alg-1/Documents/machine/')

# Load the datasets
co = pd.read_csv('comal.csv')

# Convert the date columns to datetime objects
co['date'] = pd.to_datetime(co['datetime'])

# Sort the data by date
co = co.sort_values('datetime')

# Interpolate missing data
co['Discharge'] = co['Discharge'].interpolate()

# Normalize the data
co_scaler = MinMaxScaler(feature_range=(0, 1))
co_scaled_data = co_scaler.fit_transform(co['Discharge'].values.reshape(-1, 1))

# Create the training datasets
co_train_data = co_scaled_data[:int(0.8 * len(co))]
co_x_train = []
co_y_train = []
for i in range(60, len(co_train_data)):
    co_x_train.append(co_train_data[i-60:i, 0])
    co_y_train.append(co_train_data[i, 0])
co_x_train, co_y_train = np.array(co_x_train), np.array(co_y_train)
co_x_train = np.reshape(co_x_train, (co_x_train.shape[0], co_x_train.shape[1], 1))

# Create the LSTM model for j17
co_model = Sequential()
co_model.add(LSTM(units=50, return_sequences=True, input_shape=(co_x_train.shape[1], 1)))
co_model.add(LSTM(units=50))
co_model.add(Dense(1))

# Compile the model
co_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
co_model.fit(co_x_train, co_y_train, epochs=10, batch_size=32)


# Create the test dataset
co_test_data = co_scaled_data[int(0.8 * len(co))-60:]
co_x_test = []
for i in range(60, len(co_test_data)):
    co_x_test.append(co_test_data[i-60:i, 0])
co_x_test = np.array(co_x_test)
co_x_test = np.reshape(co_x_test, (co_x_test.shape[0], co_x_test.shape[1], 1))

# Make predictions using the test dataset
co_predictions = co_model.predict(co_x_test)
co_predictions = co_scaler.inverse_transform(co_predictions)

from sklearn.metrics import mean_squared_error
import math

# Calculate the RMSE
co_actual = co_scaler.inverse_transform(co_test_data[60:, 0].reshape(-1, 1))
co_rmse = math.sqrt(mean_squared_error(co_actual, co_predictions))
print('RMSE:', co_rmse)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(co['date'].values[int(0.8 * len(co)):-(60)], 
         co_scaler.inverse_transform(co_scaled_data[int(0.8 * len(co)):-(60)]), 
         label='Actual')
plt.plot(co['date'].values[int(0.8 * len(co)):int(0.8 * len(co)) + len(co_predictions)], 
         np.squeeze(co_predictions), label='Predicted')
plt.legend()
plt.show()