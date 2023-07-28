import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv('Microsoft_Stock.csv')
df['Date'] = pd.to_datetime(df['Date'])

dates = df['Date'].values
close_session  = df['Close'].values
open_session = df['Open'].values
 
plt.figure(figsize=(15,5))
plt.plot(dates, open_session)
plt.plot(dates, close_session)

# info data
plt.title('Open vs Close',
          fontsize=20)
plt.ylabel('Price')
plt.xlabel('Periode')
plt.legend(['Open', 'Close'], loc='upper left')

# implement min max scaler 
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(close_session.reshape(-1,1))
close_session = min_max_scaler.transform(close_session.reshape(-1,1))

threshold_mae = (close_session.max() - close_session.min()) * 10/100

# since open and close are same, we will use close only
# split data validation into 20%
# Calculate the index where the last 20% starts
validation_start_idx = int(len(dates) * 0.8)

# Split the data into training and validation sets
X_train, X_validation = dates[:validation_start_idx], dates[validation_start_idx:]
y_train, y_validation = close_session[:validation_start_idx], close_session[validation_start_idx:]
print('X_train shape:', X_train.shape)
print('X_validation shape:', X_validation.shape)
print('y_train shape:', y_train.shape)
print('y_validation shape:', y_validation.shape)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

# Create windowed datasets for training and validation
train_dataset = windowed_dataset(y_train, window_size=60, batch_size=100, shuffle_buffer=1000)
validation_dataset = windowed_dataset(y_validation, window_size=60, batch_size=100, shuffle_buffer=1000)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(60, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define the optimizer with a learning rate of 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(loss=tf.keras.losses.Huber(), 
              optimizer=optimizer,
              metrics=["mae"])

num_epochs = 100

# Train the model
history = model.fit(train_dataset, epochs=num_epochs, batch_size=64,validation_data=validation_dataset,verbose=1)