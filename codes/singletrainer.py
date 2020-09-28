import math
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# plt.style.use("fivethirtyeight")

class SingleTrainer():
	"""This class loads and trains using neural networks models inputed variables"""
	def __init__(self, condition='temperature', city='Jerusalem', epochs=1, datasamples=0):
		self.condition=condition
		self.city=city
		self.datasamples=int(datasamples)
		self.epochs=epochs


	"""Set the condtion of you want train for network to use"""
	def set_condition(self, condition):
		self.condition=condition

	def set_city(self, city):
		self.city=city

	def set_epochs(epochs):
		self.epochs=epochs

	def set_samplesize(self, samples):
		self.datasamples=int(samples)

	def get_samplesize(self):
		return self.datasamples

	def train(self):

		filehandler=open(f"../datasets/data/{self.condition}.csv", "r")
		df=pd.read_csv(filehandler, index_col=[0], usecols=["datetime",self.city])

		if self.datasamples=="":
			self.set_samplesize(len(df))
			self.datasamples=self.get_samplesize()

		df=df[1:self.datasamples]

		filehandler.close()

		mean=np.mean(df)
		df=df.fillna(value=mean)

		# print(df)

		# data=df.filter([self.city])
		data=df
		self.dataframe=data
		# print(data)
		dataset=data.values
		training_data_len=math.ceil(len(dataset)*0.8)

		#scaling of data
		scaler=MinMaxScaler(feature_range=(0,1))
		scaled_data=scaler.fit_transform(dataset)
		# scaled_data

		training_data=scaled_data[0: training_data_len, :]
		#split data into x_train and y_train datasets
		x_train=[]
		y_train=[]

		for i in range(60, len(training_data)):
		  x_train.append(training_data[(i-60):i, 0])
		  y_train.append(training_data[i, 0])

		# convert to numpy arrays
		x_train, y_train=np.array(x_train), np.array(y_train)
		#Reshaping
		x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

		# creatinga model
		model=Sequential()
		model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
		model.add(LSTM(50, return_sequences=False))
		model.add(Dense(25))
		model.add(Dense(1))

		#compile model
		model.compile(optimizer='adam', loss='mean_squared_error')

		# train the model
		model.fit(x_train, y_train, batch_size=1, epochs=self.epochs)

		# testing data
		test_data=scaled_data[training_data_len-60: ,:]
		# create x_test
		x_test=[]
		y_test=dataset[training_data_len:,:]
		for i in range(60, len(test_data)):
		  x_test.append(test_data[i-60: i, 0])

		# converting to numpy array
		x_test=np.array(x_test)
		x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		# Get the model predictions 
		predictions=model.predict(x_test)
		predictions=scaler.inverse_transform(predictions)

		# Get the mean squared error
		rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

		myrmse=round(rmse, ndigits=4)
		model.save(f"../models/{self.city}/{self.condition}/{self.datasamples}_model_{myrmse}.h5")

		print("Mean Squared Error")
		print(rmse)

		#Get ploting data
		self.train=data[:training_data_len]
		self.valid=data[training_data_len: ]
		self.valid["Predictions"]=predictions


	def display_graph(self):

		# valid=self.valid[:500]
		
		valid=self.valid
		self.dataframe=self.dataframe[7000:]

		print(valid)

		# Visualize the data
		plt.figure(figsize=(16, 8))
		plt.title(f"{self.condition.capitalize()} distribution in {self.city.capitalize()}")
		plt.plot(self.dataframe, 'g', label="Training Data")
		plt.plot(valid[self.city], 'b', label="Valid Data")
		plt.plot(valid["Predictions"], 'r', label="Predictions")
		plt.xlabel("Dates", fontsize=18)
		plt.ylabel(f"{self.condition.capitalize()}", fontsize=18)
		plt.legend(loc="lower right")
		plt.show() 
