
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
os.chdir('C:/Users/alg-1/Documents/machine/')

# Load the datasets
sm = pd.read_csv('SanMarcos.csv')

# Convert the date columns to datetime objects
sm['date'] = pd.to_datetime(sm['datetime'])

# Sort the data by date
sm = sm.sort_values('datetime')

# Interpolate missing data
sm['Discharge'] = sm['Discharge'].interpolate()

# Normalize the data
sm_scaler = MinMaxScaler(feature_range=(0, 1))
sm_scaled_data = sm_scaler.fit_transform(sm['Discharge'].values.reshape(-1, 1))

# Create the training datasets
sm_train_data = sm_scaled_data[:int(0.8 * len(sm))]
sm_x_train = []
sm_y_train = []
for i in range(60, len(sm_train_data)):
    sm_x_train.append(sm_train_data[i-60:i, 0])
    sm_y_train.append(sm_train_data[i, 0])
sm_x_train, sm_y_train = np.array(sm_x_train), np.array(sm_y_train)
sm_x_train = np.reshape(sm_x_train, (sm_x_train.shape[0], sm_x_train.shape[1], 1))

# Create the LSTM model for j17
sm_model = Sequential()
sm_model.add(LSTM(units=50, return_sequences=True, input_shape=(sm_x_train.shape[1], 1)))
sm_model.add(LSTM(units=50))
sm_model.add(Dense(1))

# Compile the model
sm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
sm_model.fit(sm_x_train, sm_y_train, epochs=10, batch_size=32)


# Plot the results
import matplotlib.pyplot as plt
plt.plot(sm['date'].values[int(0.8 * len(sm)):-(60)], 
         sm_scaler.inverse_transform(sm_scaled_data[int(0.8 * len(sm)):-(60)]), 
         label='Actual')
plt.plot(sm['date'].values[int(0.8 * len(sm)):int(0.8 * len(sm)) + len(sm_predictions)], 
         np.squeeze(sm_predictions), label='Predicted')
plt.legend()
plt.show()


# Create the test dataset
sm_test_data = sm_scaled_data[int(0.8 * len(sm))-60:]
sm_x_test = []
for i in range(60, len(sm_test_data)):
    sm_x_test.append(sm_test_data[i-60:i, 0])
sm_x_test = np.array(sm_x_test)
sm_x_test = np.reshape(sm_x_test, (sm_x_test.shape[0], sm_x_test.shape[1], 1))

# Make predictions using the test dataset
sm_predictions = sm_model.predict(sm_x_test)
sm_predictions = sm_scaler.inverse_transform(sm_predictions)

# Calculate the RMSE
sm_actual = sm_scaler.inverse_transform(sm_test_data[60:, 0].reshape(-1, 1))
sm_rmse = math.sqrt(mean_squared_error(sm_actual, sm_predictions))
print('RMSE:', sm_rmse)

