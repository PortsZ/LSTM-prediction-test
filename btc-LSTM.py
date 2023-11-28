import pandas as pd
import numpy as np
import math
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib import style
import nasdaqdatalink

style.use('ggplot')

# Load API key and get the data
nasdaqdatalink.read_key(filename="./.key")
data = nasdaqdatalink.get_table('QDL/BITFINEX', code='BTCUSD')
data = data.sort_values('date')

# Feature Engineering
data['HL_PCT'] = (data['high'] - data['last']) / data['last'] * 100.0
data['PCT_daily_change'] = data['last'].pct_change() * 100
data['ma_10'] = data['last'].rolling(window=10).mean()
data['Dollar_Volume'] = data['volume'] * data['last']
data = data[['date', 'last', 'HL_PCT', 'PCT_daily_change', 'Dollar_Volume', 'ma_10']]

# Convert 'date' to datetime and set as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Fill missing values
data.fillna(-99999, inplace=True)

# Scaling the features and target separately
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = feature_scaler.fit_transform(data[['HL_PCT', 'PCT_daily_change', 'Dollar_Volume', 'ma_10']])
scaled_target = target_scaler.fit_transform(data[['last']])

# Combine scaled features and target
scaled_data = np.hstack((scaled_target, scaled_features))

# Function to create dataset for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps), :]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Prepare the dataset
time_steps = 1
X, y = create_dataset(scaled_data, scaled_data[:, 0], time_steps)

# Splitting the data
training_data_len = math.ceil(len(X) * 0.5)
X_train = X[:training_data_len]
X_test = X[training_data_len:]
y_train = y[:training_data_len]
y_test = y[training_data_len:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Define the 'train' DataFrame
train = data[:training_data_len]

# Predicting
predictions = model.predict(X_test)

# Inverse transform predictions
predictions = target_scaler.inverse_transform(predictions)  # Scaling back to original scale for 'last' price

# Define the 'valid' DataFrame: This should be the same length as 'predictions'
valid = data[training_data_len + time_steps:]  # Adjust this as necessary
valid = valid.iloc[:len(predictions)]  # Ensure valid and predictions are the same length

# Ensure the length of predictions matches the validation set
if len(predictions) != len(valid):
    raise ValueError(f"Length of predictions ({len(predictions)}) does not match length of validation data ({len(valid)})")

# Assigning predictions to the 'valid' DataFrame
valid['Predictions'] = predictions

actual = valid['last']
predicted = valid['Predictions']

# # Calculate metrics
# mae = mean_absolute_error(actual, predicted)
# mse = mean_squared_error(actual, predicted)
# rmse = np.sqrt(mse)
# r2 = r2_score(actual, predicted)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"Coefficient of Determination (R² Score): {r2}")

# # Plotting against trained data
# plt.figure(figsize=(16, 8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Last Price', fontsize=18)
# plt.plot(train['last'])
# plt.plot(valid[['last', 'Predictions']])
# plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
# plt.show()

# Define the number of steps you want to predict into the future
forecast_steps = 30

# Start with the last part of your training data as the initial input
last_sequence = scaled_data[-time_steps:]

# Store the predictions
future_predictions = []

# Predict iteratively
for _ in range(forecast_steps):
    # Get the last sequence
    current_sequence = last_sequence[-time_steps:]
    
    # Reshape to fit the model input format
    current_sequence = np.reshape(current_sequence, (1, current_sequence.shape[0], current_sequence.shape[1]))
    
    # Predict the next step and append
    next_step_prediction = model.predict(current_sequence)
    future_predictions.append(target_scaler.inverse_transform(next_step_prediction)[0, 0])
    
    # Update the sequence to include the prediction
    new_step = np.hstack((next_step_prediction, np.zeros((1, scaled_features.shape[1]))))  # Assuming other features remain zero
    last_sequence = np.vstack((last_sequence, new_step))

# Convert future predictions to a DataFrame for easier plotting
future_dates = pd.date_range(start=data.index[-1], periods=forecast_steps)
future_df = pd.DataFrame(index=future_dates, data={'Forecast': future_predictions})

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(data['last'], label='Historical Daily Closing Price')
plt.plot(future_df['Forecast'], label='Future Predictions')
plt.title('Extended Future Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
