from multipletrainer import MultipleTrainer
import output_trainer
from singletrainer import SingleTrainer

def train_multiple():
	"""Trainfor temperature, pressure, humidity, windspeed and wind directions"""
	multipletrainer=MultipleTrainer(display_graphs=True)
	multipletrainer.set_cities(["Jerusalem"])
	multipletrainer.set_conditions(['pressure','wind_speed','wind_direction'])
	multipletrainer.set_samplesize(10000)
	multipletrainer.train_multiple()

def outputTest():
	out=output_trainer.OutputTrainer()
	out.create_patterns()
	# out.set_patterns({"sky is clear":0,'haze':1,'moderate rain':2})
	print(out.get_created_patterns())
	print(out.get_numeric_desc_df())
	# out.set_patterns("sky is clear")
	print(out.get_description_dataframe())
	# print(out.get_numeric_desc_df())

def train_single():
	singleTrainer=SingleTrainer()
	singleTrainer.set_samplesize(10000)
	singleTrainer.train()
	singleTrainer.display_graph()


# outputTest()
# train_multiple()
train_single()