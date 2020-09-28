import math
import os
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")



class OutputTrainer():
	"""OuputTrainer class trains the system using description dataset"""
	def __init__(self, cities=["Toronto", "Houston"], conditions=['temperature','pressure'],epochs=1, datasamples=1000, display_graphs=True):
		super().__init__()
		self.conditions=conditions
		self.cities=cities
		self.epochs=epochs
		self.datasamples=datasamples
		self.display_graphs=display_graphs
		self.patterns={}
		

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

		self.set_patterns(patterns)
		print(self.get_created_patterns())

	def get_created_patterns(self):
		return self.patterns

	def set_patterns(self, keyword):
		for i in self.patterns:
			pattern_item=str(i)
			if keyword in pattern_item:
				self.patterns[i]=1
			else:
				self.patterns[i]=0

		print(self.patterns)

	def desc_df_to_numeric(self):
		self.set_patterns("sky is clear")
		self.desc_df=self.get_description_dataframe()
		for city in self.cities:
			print(self.desc_df[city])
			self.desc_df[city]=[self.patterns[pattern] for pattern in self.desc_df[city]]

	def get_numeric_desc_df(self):
		return self.desc_df

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

