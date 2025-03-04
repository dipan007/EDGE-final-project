# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset
data = pd.read_csv('mydata.csv')  # Replace 'mydata.csv' with your dataset filename

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Step 2: Select data for a specific station (e.g., Dhaka)
station_data = data[data['Station'] == 'Dhaka'][['Year', 'Month', 'Rainfall']]

# Create a date column for time series analysis
station_data['Date'] = pd.to_datetime(station_data['Year'].astype(str) + '-' + station_data['Month'].astype(str))
station_data.set_index('Date', inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# Plot the rainfall data over time
plt.figure(figsize=(14, 6))
plt.plot(station_data.index, station_data['Rainfall'], label='Rainfall in Dhaka', color='blue')
plt.title('Monthly Rainfall Over Time in Dhaka (1990–2016)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Normalize the rainfall data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(station_data[['Rainfall']])

# Step 5: Prepare time series data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Set the time step (e.g., 12 months for yearly seasonality)
time_step = 12
X, y = create_dataset(scaled_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 6: Split the data into training and testing sets
train_size = int(len(X) * 0.8)  # 80% training, 20% testing
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Step 7: Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))  # First LSTM layer
model.add(LSTM(50, return_sequences=False))  # Second LSTM layer
model.add(Dense(25))  # Dense layer
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 8: Train the model
print("\nTraining the LSTM model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Step 9: Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 10: Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
train_mae = mean_absolute_error(y_train, train_predict))
test_mae = mean_absolute_error(y_test, test_predict))

print(f"\nTrain RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# Step 11: Plot the results
plt.figure(figsize=(14, 6))
plt.plot(station_data.index[time_step:train_size + time_step], y_train, label='Actual Train Data', color='blue')
plt.plot(station_data.index[time_step:train_size + time_step], train_predict, label='Predicted Train Data', color='orange')
plt.plot(station_data.index[train_size + time_step:], y_test, label='Actual Test Data', color='green')
plt.plot(station_data.index[train_size + time_step:], test_predict, label='Predicted Test Data', color='red')
plt.title('Rainfall Forecasting Using LSTM (1990–2016)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)
plt.show()