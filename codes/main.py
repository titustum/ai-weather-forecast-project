import math
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Flatten
from tensorflow.keras.models import Sequential ,load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

filehandler=open("../datasets/data/weather_description.csv", "r")

desc_df=pd.read_csv(filehandler, index_col=[0], usecols=["datetime", "Toronto"])
filehandler.close()

desc_df =desc_df[1:]

patterns={'sky is clear': 0, 'scattered clouds': 1, 'overcast clouds': 2, 'few clouds': 3, 'haze': 4, 'broken clouds': 5, 'light intensity drizzle': 6, 'mist': 7, 'fog': 8, 'light intensity shower rain': 9, 'light rain': 10, 'proximity shower rain': 11, 'moderate rain': 12, 'heavy shower snow': 13, 'heavy snow': 14, 'light snow': 15, 'heavy intensity rain': 16, 'thunderstorm with light rain': 17, 'snow': 18, 'shower rain': 19, 'shower snow': 20, 'light shower sleet': 21, 'freezing rain': 22, 'thunderstorm': 23, 'thunderstorm with rain': 24, 'heavy intensity shower rain': 25, 'thunderstorm with heavy rain': 26, 'proximity thunderstorm': 27, 'very heavy rain': 28, 'light shower snow': 29, 'light rain and snow': 30, 'light intensity drizzle rain': 31, 'squalls': 32, 'drizzle': 33, 'rain and snow': 34}

def patterns_occurances():
	mypat={}
	for occ in desc_df.Toronto:
		if occ in mypat:
			mypat[occ]=mypat[occ]+1
		else:
			mypat[occ]=1
	return mypat

print(patterns_occurances())



def set_pattern(keyword):
	for i in patterns:
		pattern_item=str(i)
		if keyword in pattern_item:
			patterns[i]=1
		else:
			patterns[i]=0
	return patterns

# occurences={'sky is clear': 13914, 'scattered clouds': 4499, 'overcast clouds': 6252, 'few clouds': 2743, 'haze': 318, 'broken clouds': 4868, 'light intensity drizzle': 187, 'mist': 2650, 'fog': 251, 'light intensity shower rain': 703, 'light rain': 3355, 'proximity shower rain': 263, 'moderate rain': 1393, 'heavy shower snow': 170, 'heavy snow': 317, 'light snow': 1827, 'heavy intensity rain': 473, 'thunderstorm with light rain': 65, 'snow': 314, 'shower rain': 17, 'shower snow': 6, 'light shower sleet': 12, 'freezing rain': 3, 'thunderstorm': 42, 'thunderstorm with rain': 14, 'heavy intensity shower rain': 9, 'thunderstorm with heavy rain': 11, 'proximity thunderstorm': 62, 'very heavy rain': 41, 'light shower snow': 449, 'light rain and snow': 1, 'light intensity drizzle rain': 16, 'squalls': 2, 'drizzle': 4, 'rain and snow': 1}


patterns=set_pattern("light intensity shower rain")

# print(patterns)


desc_df.Toronto=[patterns[pattern] for pattern in desc_df.Toronto]

def create_dataframe(condition, city):
	filehandler=open(f"../datasets/data/{condition}.csv", "r")
	df=pd.read_csv(filehandler, index_col=[0], usecols=["datetime", city])
	filehandler.close()
	mean=np.mean(df)
	df=df.fillna(value=mean)
	return df[1:]


temperature_df=create_dataframe("temperature", "Toronto")
humidity_df=create_dataframe("humidity", "Toronto")
pressure_df=create_dataframe("pressure", "Toronto")
windspd_df=create_dataframe("wind_speed", "Toronto")
winddir_df=create_dataframe("wind_direction", "Toronto")

#Use one list variable to join all to one dataframe
desc_df["temperature"]=temperature_df["Toronto"].values
desc_df["pressure"]=pressure_df["Toronto"].values
desc_df["humidity"]=humidity_df["Toronto"].values
desc_df["winddir"]=winddir_df["Toronto"].values
desc_df["windspd"]=windspd_df["Toronto"].values
desc_df["Description"]=desc_df["Toronto"].values

# print(desc_df[20:])

dataset=desc_df

X_data=dataset[{"temperature","pressure", "humidity","windspd", "winddir"}]
y_data=dataset["Description"]

scaler=MinMaxScaler(feature_range=(0,1))

#Training data
training_data_len=int(math.ceil(len(dataset)*0.7))
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

def ann_trainer():
	model=Sequential()
	model.add(Dense(5, activation="relu"))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(2, activation="softmax"))

	model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")
	model.fit(X_train_final, y_train_final, batch_size=1, epochs=1)

	test_loss, test_acc = model.evaluate(X_test, y_test)
	print("ANN classification accuracy: ", test_acc)



def naiveb_trainer():
	gn=GaussianNB()
	model=gn.fit(X_train_final, y_train_final)
	preds=model.predict(X_test)
	accuracy=accuracy_score(y_test, preds)
	print("Bayes classifier accuracy: ", accuracy)
	print("\n\n X_test data")
	print(y_test)

	print("\n Predicted data")
	print(preds)

ann_trainer()
naiveb_trainer()
