import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
os.chdir('C:/Users/alg-1/Documents/machine/')

# Load the datasets
j17 = pd.read_csv('j17.csv')

# Convert the date columns to datetime objects
j17['date'] = pd.to_datetime(j17['datetime'])

# Sort the data by date
j17 = j17.sort_values('date')

# Interpolate missing data
j17['waterle'] = j17['waterle'].interpolate()

# Normalize the data
j17_scaler = MinMaxScaler(feature_range=(0, 1))
j17_scaled_data = j17_scaler.fit_transform(j17['waterle'].values.reshape(-1, 1))

# Create the training datasets
j17_train_data = j17_scaled_data[:int(0.8 * len(j17))]
j17_x_train = []
j17_y_train = []
for i in range(60, len(j17_train_data)):
    j17_x_train.append(j17_train_data[i-60:i, 0])
    j17_y_train.append(j17_train_data[i, 0])
j17_x_train, j17_y_train = np.array(j17_x_train), np.array(j17_y_train)
j17_x_train = np.reshape(j17_x_train, (j17_x_train.shape[0], j17_x_train.shape[1], 1))

# Create the LSTM model for j17
j17_model = Sequential()
j17_model.add(LSTM(units=50, return_sequences=True, input_shape=(j17_x_train.shape[1], 1)))
j17_model.add(LSTM(units=50))
j17_model.add(Dense(1))

# Compile the model
j17_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
j17_model.fit(j17_x_train, j17_y_train, epochs=10, batch_size=32)

# Create the test dataset
j17_test_data = j17_scaled_data[int(0.8 * len(j17)):-(60)]
j17_x_test = []
for i in range(60, len(j17_test_data)):
    j17_x_test.append(j17_test_data[i-60:i, 0])
j17_x_test = np.array(j17_x_test)
j17_x_test = np.reshape(j17_x_test, (j17_x_test.shape[0], j17_x_test.shape[1], 1))

# Make predictions using the test dataset
j17_predictions = j17_model.predict(j17_x_test)
j17_predictions = j17_scaler.inverse_transform(j17_predictions)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(j17['date'].values[int(0.8 * len(j17)):-(60)], 
         j17_scaler.inverse_transform(j17_scaled_data[int(0.8 * len(j17)):-(60)]), 
         label='Actual')
plt.plot(j17['date'].values[int(0.8 * len(j17)):int(0.8 * len(j17)) + len(j17_predictions)], 
         np.squeeze(j17_predictions), label='Predicted')
plt.legend()
plt.show()


# Make predictions using the test dataset
j17_predictions = j17_model.predict(j17_x_test)
j17_predictions = j17_scaler.inverse_transform(j17_predictions)

# Calculate the RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(j17_scaler.inverse_transform(j17_test_data), j17_predictions))
print('RMSE:', rmse)

