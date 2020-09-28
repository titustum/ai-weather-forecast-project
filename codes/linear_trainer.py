import math
import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

df=pd.read_csv("../datasets/data/temperature.csv", index_col=[0])
df=df[1:20000]
# df=df[1:44461] #for production

df = df.apply(pd.to_numeric, errors='coerce')


data=df.filter(["Toronto"])

dataset=df.values

training_data_len=int(math.ceil(len(dataset)*0.8))
training_data=dataset[0:training_data_len, :]
# for i in dataset[0]:
# 	print(i)

x_train=[]
y_train=[]

# print(training_data[0:61, 0])

for i in range(60, training_data_len):
	x_train.append(training_data[(i-60):i, 0])
	y_train.append(training_data[i,0])



reg=linear_model.LinearRegression()
reg.fit(x_train, y_train)
print(reg.coef_)

test_data=dataset[training_data_len:]

x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60, len(test_data)):
	x_test.append(test_data[i-60:i, 0])


# print(len(y_test))
# print(y_test)
# Predict
predictions=reg.predict(x_test)
# print(predictions)
# print(len(predictions))


valid=data[training_data_len:len(dataset)-60]
valid["predictions"]=predictions
print(valid)

print(np.mean(valid["predictions"]))
print(np.mean(valid["Toronto"]))

print("Mean_squared_error")
rmse=np.sqrt( np.mean(valid["predictions"]-valid["Toronto"])**2 )
print(rmse)

valid=valid[:500]

plt.figure(figsize=(16,8))
# plt.plot(x_test)
plt.plot(valid[["Toronto", "predictions"]])
plt.xlabel("Dates")
plt.ylabel("Temperature (Pascals)")
plt.legend(["Real temperature", "Predicted temperature"])
plt.title("Graph of Temperature Distribution and Prediction in Toronto, Canada")
plt.show()