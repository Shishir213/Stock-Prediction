import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st



yf.pdr_override()
start = '2008-01-01'
end = '2022-03-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')


stock_data = pdr.get_data_yahoo(user_input, start, end)

#Describing Data
st.subheader('Data from 2008 - 2022')
st.write(stock_data.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = stock_data.Close.rolling(100).mean
fig = plt.figure(figsize = (12, 6))
plt.plot(stock_data.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = stock_data.Close.rolling(100).mean
ma200 = stock_data.Close.rolling(200).mean
fig = plt.figure(figsize = (12, 6))
plt.plot(stock_data.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)


#Splitting data into Training and Testing



#Creating Dataframe for Training and Testing
data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data)*0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70):int(len(stock_data))])

#Scaling the Data from 0 and 1 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)






#Loading the model
model = load_model('keras_model.h5')

#Create the data testing set
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

#Transform the data into 0 to 1
input_data = scaler.fit_transform(final_df)

#Define xTest and yTest
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

#Convert into array
x_test, y_test = np.asarray(x_test), np.asarray(y_test)

#Making Predictions
y_predicted = model.predict(x_test)
y_predicted.shape

#Scaling up the values
scaler = scaler.scale_
scaler_factor = 1/scaler[0]

y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor



#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
