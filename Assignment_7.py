import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
data = pd.read_csv('GOOG.csv', date_parser = True)
data.tail()

data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(data_training)
training_data 

X_train = []
y_train = []
for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.2))


model.add(Dense(units = 1))

model.compile(optimizer='adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

data_test.head()

data_training.tail(60)

data_training.tail(60)

df = past_60_days.append(data_test, ignore_index = True,sort=True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

inputs = scaler.transform(df)
inputs

X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

y_pred = model.predict(X_test)
scaler.scale_
scale = 1/8.18605127e-04
scale

y_pred = y_pred*scale
y_test = y_test*scale

plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()