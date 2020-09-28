import math
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

class OutputTrainer():
	"""OuputTrainer class trains the system using description dataset"""
	def __init__(self, cities=["Toronto", "Houston"], conditions=['temperature','pressure', 'humidity', 'wind_speed'],epochs=1, datasamples=1000, display_graphs=True):
		super().__init__()
		self.conditions=conditions
		self.cities=cities
		self.epochs=epochs
		self.datasamples=datasamples
		self.display_graphs=display_graphs

	def set_conditions(self, conditions):
		self.conditions=conditions

	def set_cities(self, cities):
		self.cities=cities

	def set_epochs(epochs):
		self.epochs=epochs

	def set_samplesize(self, samples):
		self.datasamples=int(samples)

	def get_samplesize(self):
		return self.datasamples

	def set_patterns(self, pattern):
		self.patterns=pattern

	def create_patterns(self):
		patterns={}
		count=0
		desc_dataframe=self.get_description_dataframe()
		for city in self.cities:
			for x in desc_dataframe[city]:
				if x in patterns:
					continue
				else:
					patterns[x]=count
					count=count+1

		self.patterns=patterns

		# print(patterns)
		# self.set_patterns(patterns)
		# print(self.get_created_patterns())

	def set_patterns(self, keyword):
		self.keyword=keyword
		for i in self.patterns:
			pattern_item=str(i)
			if keyword in pattern_item:
				self.patterns[i]=1
			else:
				self.patterns[i]=0

		# print(self.patterns)

	def desc_df_to_numeric(self):
		# self.set_patterns("sky is clear")
		self.desc_df=self.get_description_dataframe()
		for city in self.cities:
			# print(self.desc_df[city])
			self.desc_df[city]=[self.patterns[pattern] for pattern in self.desc_df[city]]

	def get_numeric_desc_df(self):
		return self.desc_df

	def get_created_patterns(self):
		return self.patterns


	def get_description_dataframe(self):
		return self.get_dataframe("weather_description")

	def get_dataframe(self, condition):
		filehandler=open(f"../datasets/data/{condition}.csv", "r")
		usable_cols=[]
		usable_cols.append("datetime")
		for x in self.cities:
			usable_cols.append(x)
		df=pd.read_csv(filehandler, index_col=[0], usecols=usable_cols)
		filehandler.close()
		mean=np.mean(df)
		df=df.fillna(value=mean)
		return df[1:]

	def train(self):
		temp_df=self.get_dataframe("temperature")
		press_df=self.get_dataframe("pressure")
		hum_df=self.get_dataframe("humidity")
		winddir_df=self.get_dataframe("wind_direction")
		windspd_df=self.get_dataframe("wind_speed")

		for city in self.cities:
			print(f"\n\n=======================Training for {city.capitalize()}=======================================\n\n")
			
			self.city=city

			desc_df=self.desc_df.filter([city])
			desc_df["temperature"]=temp_df[city].values
			desc_df["pressure"]=press_df[city].values
			desc_df["humidity"]=hum_df[city].values
			desc_df["wind_speed"]=windspd_df[city].values
			desc_df["wind_direction"]=winddir_df[city].values
			desc_df["targets"]=desc_df[city].values

			X_train, y_train, X_test, y_test=self.process_dataframe(desc_df)
			self.create_neural_model()
			self.train_neural_model(X_train, y_train, X_test, y_test)
			self.create_naive_bayes_model(X_train, y_train, X_test, y_test)



	def process_dataframe(self, dataframe):
		dataset=dataframe
		X_data=dataframe.filter(self.conditions)
		y_data=dataframe["targets"]

		scaler=MinMaxScaler(feature_range=(0,1))

		#Training data
		training_data_len=int(math.ceil(len(dataset)*0.8))
		X_train=X_data[0:training_data_len]
		y_train=y_data[0:training_data_len]

		X_train_scaled=scaler.fit_transform(X_train.values)

		X_train_final=np.array(X_train_scaled)
		print(X_train_final.shape)
		y_train_final=np.array(y_train).reshape(-1, 1)
		print(y_train_final.shape)

		 # Testing data
		X_test= X_data[training_data_len:]
		y_test=y_data[training_data_len:]

		X_test=scaler.fit_transform(X_test)

		X_test=np.array(X_test)
		y_test=np.array(y_test).reshape(-1, 1)	

		return X_train_final, y_train_final, X_test, y_test

	def create_neural_model(self):
		model=Sequential()
		model.add(Dense(5, activation="relu"))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(2, activation="softmax"))

		model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
		return model

	def train_neural_model(self, X, y, X_test, y_test):
		model=self.create_neural_model()
		model.fit(X,y, batch_size=1, epochs=1)		
		test_loss, test_acc = model.evaluate(X_test, y_test)

		accuracy=test_acc
		accuracy=round(accuracy, ndigits=4)
		if not os.path.exists(f"../models/exit_neural_models/{self.city}/{self.keyword}/"):
			os.makedirs(f"../models/exit_neural_models/{self.city}/{self.keyword}/")
		model.save(f"../models/exit_neural_models/{self.city}/{self.keyword}/{self.datasamples}_samples_{accuracy}_accuracy.h5")

		print("\n ANN classification accuracy: ", test_acc)
		print("\n")

	def create_naive_bayes_model(self, X, y, X_test, y_test):
		gn=GaussianNB()
		model=gn.fit(X, y)
		preds=model.predict(X_test)
		accuracy=accuracy_score(y_test, preds)

		accuracy=round(accuracy, ndigits=4)
		if not os.path.exists(f"../models/exit_bayes_models/{self.city}/{self.keyword}/"):
			os.makedirs(f"../models/exit_bayes_models/{self.city}/{self.keyword}/")

		# with open(f"../models/exit_bayes_models/{self.city}/{self.keyword}/{self.datasamples}_samples_{accuracy}_accuracy.pickle", "w") as f:
		# 	pickle.dump(model, f)
		# f.close()

		print("\n Bayes classifier accuracy: ", accuracy)
		print("\n")






patt={'sky is clear': 0, 'scattered clouds': 1, 'overcast clouds': 2, 'few clouds': 3, 'haze': 4, 'broken clouds': 5, 'light intensity drizzle': 6, 'mist': 7, 'fog': 8, 'light intensity shower rain': 9, 'light rain': 10, 'proximity shower rain': 11, 'moderate rain': 12, 'heavy shower snow': 13, 'heavy snow': 14, 'light snow': 15, 'heavy intensity rain': 16, 'thunderstorm with light rain': 17, 'snow': 18, 'shower rain': 19, 'shower snow': 20, 'light shower sleet': 21, 'freezing rain': 22, 'thunderstorm': 23, 'thunderstorm with rain': 24, 'heavy intensity shower rain': 25, 'thunderstorm with heavy rain': 26, 'proximity thunderstorm': 27, 'very heavy rain': 28, 'light shower snow': 29, 'light rain and snow': 30, 'light intensity drizzle rain': 31, 'squalls': 32, 'drizzle': 33, 'rain and snow': 34, 'proximity moderate rain': 35, 'dust': 36, 'smoke': 37, 'proximity thunderstorm with rain': 38, 'thunderstorm with light drizzle': 39, 'thunderstorm with drizzle': 40, 'sand': 41, 'heavy intensity drizzle': 42, 'volcanic ash': 43}


# c=0  
# for x in patt:
# 	print(f"\n\n=============Now training for {x}================\n\n")
# 	out=OutputTrainer()
# 	out.set_cities(["Toronto","Houston","Jerusalem"])
# 	out.create_patterns()
# 	out.set_samplesize(10000)
# 	out.set_patterns(x)
# 	out.desc_df_to_numeric()
# 	# print(out.get_numeric_desc_df())
# 	out.train()
# 	c+=1
# 	if c==10:
# 		break
# # print(out.get_created_patterns())
# # print(out.get_description_dataframe())



class UseModels(object):
	"""docstring for UseModels"""
	def __init__(self, city='Toronto'):
		super(UseModels, self).__init__()
		self.city = city


	def create_patterns(self):
		self.cities=["Toronto"]
		patterns={}
		count=0
		desc_dataframe=self.get_description_dataframe()
		for city in self.cities:
			for x in desc_dataframe[city]:
				if x in patterns:
					continue
				else:
					patterns[x]=count
					count=count+1

		self.patterns=patterns

	def set_patterns(self, keyword):
		self.keyword=keyword
		for i in self.patterns:
			pattern_item=str(i)
			if keyword in pattern_item:
				self.patterns[i]=1
			else:
				self.patterns[i]=0

	def desc_df_to_numeric(self):
		self.desc_df=self.get_description_dataframe()
		self.desc_df[self.city]=[self.patterns[pattern] for pattern in self.desc_df[self.city]]

	def get_numeric_desc_df(self):
		return self.desc_df

	def get_created_patterns(self):
		return self.patterns

	def get_description_dataframe(self):
		return self.get_dataframe("weather_description")
	
	def get_dataframe(self, condition):
		filehandler=open(f"../datasets/data/{condition}.csv", "r")
		df=pd.read_csv(filehandler, index_col=[0], usecols=["datetime", self.city])
		filehandler.close()
		mean=np.mean(df)
		df=df.fillna(value=mean)
		return df[1:]

	def get_x_test(self, dataframe):
		dataset=dataframe.filter(self.city)
		training_data_len=int(math.ceil(len(dataframe)*0.8))

		X_test=[]
		y_test=dataset[training_data_len:]
		for i in range(60, len(y_test)):
		  X_test.append(y_test[i-60: i])

		scaler=MinMaxScaler(feature_range=(0,1))

		X_test=scaler.fit_transform(X_test)
		X_test=np.array(X_test)
		y_test=np.array(y_test).reshape(-1, 1)	
		return X_test, y_test

	def set_desc_dataset(self):

		self.desc_df=self.get_description_dataframe()

		temp_df=self.get_dataframe("temperature")
		press_df=self.get_dataframe("pressure")
		hum_df=self.get_dataframe("humidity")
		winddir_df=self.get_dataframe("wind_direction")
		windspd_df=self.get_dataframe("wind_speed")


		desc_df=self.desc_df.filter([self.city])
		desc_df["temperature"]=temp_df[self.city].values
		desc_df["pressure"]=press_df[self.city].values
		desc_df["humidity"]=hum_df[self.city].values
		desc_df["wind_speed"]=windspd_df[self.city].values
		desc_df["wind_direction"]=winddir_df[self.city].values
		desc_df["targets"]=desc_df[self.city].values
		self.newdesc=desc_df

	def get_desc_dataset(self):
		return self.newdesc

	def get_temperature_preds(self):
		model=load_model(f"../models/{self.city}/temperature/10000_samples_1.3525_error.h5")
		temp_df=self.get_dataframe("temperature")
		X_test, y_test=self.get_x_test(temp_df)

		predicts=model.predict(X_test)
		return predicts


	def get_all_preds(self):
		conditions=["temperature","pressure","humidity","wind_speed","wind_direction"]
		models=[""]
		for condition in conditions:
			X_test, y_test=self.get_x_test(self.newdesc)
		# model=load_model(f"../models/{self.city}/{self.keyword}/{self.datasamples}_samples_{accuracy}_accuracy.h5")

