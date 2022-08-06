# import required packages
import pandas
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from utils import *

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# *************************************************
	#			Main-Specific Data
	# *************************************************
	# No. of features to consider from each data row
	No_features = 4

	# No. of previous days to consider for current predictions
	No_rel_days = 3
	
		
	# *************************************************
	#		1. Load your saved model
	# *************************************************
	rnn_model = None
	if (No_rel_days == 3): 
		rnn_model = models.load_model("./models/20986041_RNN_model.h5")
	elif (No_rel_days == 6):
		rnn_model = models.load_model("./models/6days_20986041_RNN_model.h5")
	
	
	
	# *************************************************
	#		2. load your testing data
	# *************************************************
	read = None
	if (No_rel_days == 3): 
		read = pandas.read_csv('./data/test_data_RNN.csv')
	elif (No_rel_days == 6):
		read = pandas.read_csv('./data/6days_test_data_RNN.csv')
	
	label = read['Open_Current_Day']
	input = read.drop(['Open_Current_Day', 'Date'], axis=1)
	dates = read['Date']

	# -------- Preprocessing -------
	# Scale to a range of (0,1)
	sc = MinMaxScaler(feature_range=(0, 1))
	input = np.array(sc.fit_transform(input))
	label = np.array(label)

	# Reshape the input to fit the requirements of the Recurrent Neural Network (RNN)
	# Required Format - [batch, timesteps, feature] 
	# ..... in our case is [size(input), No. of prev. days , No. of features]
	struc_data = np.zeros((input.shape[0], No_rel_days, No_features))
	for i in range(len(input)):
		row = np.zeros((No_rel_days, No_features))
		for j in range(No_rel_days):
			row[j] = np.array([input[i][j + (0 * No_rel_days)],
							input[i][j + (1 * No_rel_days)],
							input[i][j + (2 * No_rel_days)],
							input[i][j + (3 * No_rel_days)]])

		struc_data[i] = row.reshape(row.shape[0], row.shape[1])

	test_input = struc_data
	test_label = label


	# *************************************************
	# 3. Run prediction on the test data and output required plot and loss
	# *************************************************
	results = rnn_model.predict(test_input)

	# Report Loss
	print("Loss = " + str(mean_squared_error(results, test_label)))

	# Plot Prediction Graph
	dates = pandas.to_datetime(dates)
	predictions = pandas.DataFrame(results, columns = ["Stock Value Predictions"])
	real = pandas.DataFrame(test_label, columns = ["Real Stock Values"])
	
	final_dataframe = pandas.concat([dates, real, predictions], axis=1)
	final_dataframe = final_dataframe.sort_values(by = "Date")
	plot = final_dataframe.plot(x = "Date", y = ["Stock Value Predictions", "Real Stock Values"])
	plot.set_ylabel("Stock Value")
	pyplot.show()
	