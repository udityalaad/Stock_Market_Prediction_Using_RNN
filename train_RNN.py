# import required packages
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from utils import *

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow



# -------------------------------------------------------------------------------------------------
#           Function -to- Prepare the Test and Train Data - From Given File
# -------------------------------------------------------------------------------------------------
def prepare_date (No_features, No_rel_days):
	# -------- Read Given File and Create New Cumulative Tables --------
	# Read given file
	all_data = pandas.read_csv('./data/q2_dataset.csv')
	print(all_data)

	# Lists to store the input and label data in required form
	input_data = np.zeros((len(all_data) - No_rel_days, No_features * No_rel_days))
	label_data = np.zeros((len(all_data) - No_rel_days, 1))
	date_list = []

	# Populate the new arrays
	for i in range(len(input_data)):
		# Previous Days' Data (from least recent -to- most recent day)
		for j in range(No_rel_days):
			day = (No_rel_days - j)
			input_data[i][j + (0 * No_rel_days)] = all_data.iloc[i + day] [2]
			input_data[i][j + (1 * No_rel_days)] = all_data.iloc[i + day] [3]
			input_data[i][j + (2 * No_rel_days)] = all_data.iloc[i + day] [4]
			input_data[i][j + (3 * No_rel_days)] = all_data.iloc[i + day] [5]
		
		# Current Days Opening
		label_data[i][0] = all_data.iloc[i][3]

		# Current Date
		date_list.append(all_data.iloc[i][0])

	dates = np.array(pandas.to_datetime(date_list), dtype=np.datetime64)

	# Create Tables
	if (No_rel_days == 3): 
		input_heading = ["Volume_Day_1", "Volume_Day_2", "Volume_Day_3",
						"Open_Day_1", "Open_Day_2", "Open_Day_3",
						"High_Day_1", "High_Day_2", "High_Day_3",
						"Low_Day_1", "Low_Day_2", "Low_Day_3"
						]
	elif (No_rel_days == 6):
		input_heading = ["Volume_Day_1", "Volume_Day_2", "Volume_Day_3", "Volume_Day_4", "Volume_Day_5", "Volume_Day_6",
						"Open_Day_1", "Open_Day_2", "Open_Day_3", "Open_Day_4", "Open_Day_5", "Open_Day_6",
						"High_Day_1", "High_Day_2", "High_Day_3", "High_Day_4", "High_Day_5", "High_Day_6",
						"Low_Day_1", "Low_Day_2", "Low_Day_3", "Low_Day_4", "Low_Day_5", "Low_Day_6"
						]

	label_heading = ["Open_Current_Day"]
	date_heading = ["Date"]

	input_table = pandas.DataFrame(input_data, columns = input_heading)
	print(input_table)
	label_table = pandas.DataFrame(label_data, columns = label_heading)
	print(label_table)
	date_table = pandas.DataFrame(dates, columns = date_heading)
	print(date_table)


	# -------- Split Train and Test Data --------
	train_input, test_input, train_label, test_label = train_test_split(input_table, label_table, test_size= (1- 70 / 100), random_state=40)
	train_input1, test_input1, train_date, test_date = train_test_split(input_table, date_table, test_size= (1- 70 / 100), random_state=40)

	# -------- Write to CSV Files --------
	train_file = pandas.concat([train_input, train_label, train_date], axis=1)
	test_file = pandas.concat([test_input, test_label, test_date], axis=1)

	if (No_rel_days == 3): 
		train_file.to_csv("data/train_data_RNN.csv", header=True, index=False)
		test_file.to_csv("data/test_data_RNN.csv", header=True, index=False)
	elif (No_rel_days == 6):
		train_file.to_csv("data/6days_train_data_RNN.csv", header=True, index=False)
		test_file.to_csv("data/6days_test_data_RNN.csv", header=True, index=False)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
#				Class for Reccurent Neural Network
#		References: 0] https://www.tensorflow.org/guide/keras/rnn
# 					1] https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
#					2] https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
#					3] https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
# -------------------------------------------------------------------------------------------------
class RNN ():
	# ---- Set-up and Compile the network ----
	def __init__ (self, model_type,loss_function, optimizer, shape, batch_size):
		self.batch_size = batch_size
		
		if model_type == 'Simple RNN':
			self.model = Sequential([
					layers.SimpleRNN(64, input_shape=shape, return_sequences=True),	# SimpleRNN Layer-1	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.SimpleRNN(128, return_sequences=True),					# SimpleRNN Layer-2	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.SimpleRNN(256),											# SimpleRNN Layer-3	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.Dense(1)													# SimpleRNN Layer 	- With no explicit activation
				])
		elif model_type == 'GRU':
			self.model = Sequential([
					layers.GRU(64, input_shape=shape, return_sequences=True),	# GRU Layer-1	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.GRU(128, return_sequences=True),						# GRU Layer-2	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.GRU(256),											# GRU Layer-3	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.Dense(1)												# Output Layer 	- With no explicit activation
				])
		elif model_type == 'LSTM':
			self.model = Sequential([
					layers.LSTM(64, input_shape=shape, return_sequences=True),	# LSTM Layer-1	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.LSTM(128, return_sequences=True),					# LSTM Layer-2	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.LSTM(256),											# LSTM Layer-3	- By default uses 'tanh' and 'sigmoid' for activation and recurrent-activation respectively
					layers.Dense(1)												# Output Layer 	- With no explicit activation
				])

		self.model.compile(loss = loss_function, optimizer = optimizer, metrics=['mae'])

	# ---- Train and test the network with given inputs ----
	def train_and_test (self, train_input, train_label, no_epochs):
		return self.model.fit(train_input, train_label, epochs = no_epochs, batch_size = self.batch_size)

	# ---- Test the network with given inputs ----
	def test (self, input, label):
		return self.model.evaluate(input, label)

	# ---- Predict Outcomes for the given Inputs ----
	def predict (self, input):
		return self.model.predict(input)
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
#					Main Function
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	# *************************************************
	#			Main-Specific Data
	# *************************************************
	# No. of features to consider from each data row
	No_features = 4

	# No. of previous days to consider for current predictions
	No_rel_days = 3

	# Optimizer
	G_optimizer = 'adam'

	# Common Loss Function
	G_loss_function = 'mean_squared_error'

	# Shape of each input
	G_input_shape = (No_rel_days, No_features)

	# No. of epochs to train for
	G_no_epochs = 250

	# Batch_Size
	G_batch_size = 16

	# *************************************************
	#		0. Prepare and Save Data to CSV Files
	# *************************************************
	# prepare_date(No_features = No_features, No_rel_days = No_rel_days)
	
		
	# *************************************************
	#		1. load your training data
	# *************************************************
	read = None
	if (No_rel_days == 3): 
		read = pandas.read_csv('./data/train_data_RNN.csv')
	elif (No_rel_days == 6):
		read = pandas.read_csv('./data/6days_train_data_RNN.csv')
	
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

	train_input = struc_data
	train_label = label


	# *************************************************
	#		2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss
	# *************************************************
	# # With Simple RNN
	# rnn = RNN(model_type = 'Simple RNN', loss_function = G_loss_function, optimizer = G_optimizer, shape = G_input_shape, batch_size=G_batch_size)
	# rnn_results = rnn.train_and_test(train_input = train_input, train_label = train_label, no_epochs = G_no_epochs)

	# # # With GRU (Gated Recurrent Units)
	# rnn = RNN(model_type = 'GRU', loss_function = G_loss_function, optimizer = G_optimizer, shape = G_input_shape, batch_size=G_batch_size)
	# rnn_results = rnn.train_and_test(train_input = train_input, train_label = train_label, no_epochs = G_no_epochs)

	# With LSTM (Long Short-Term Memory)
	rnn = RNN(model_type = 'LSTM', loss_function = G_loss_function, optimizer = G_optimizer, shape = G_input_shape, batch_size=G_batch_size)
	rnn_results = rnn.train_and_test(train_input = train_input, train_label = train_label, no_epochs = G_no_epochs)


	# Draw Plots
	g = Graphs()
	g.plot_singular(input = rnn_results, title = 'RNN Loss', plot = 'loss', x_label='Epoch', y_label='Loss')
	g.plot_singular(input = rnn_results, title = 'RNN MAE', plot = 'mae', x_label='Epoch', y_label='MAE')


	# Report Results
	print("\n--------------------------------------------------------------------------------------")
	print("After " + str(G_no_epochs) + " Epochs")
	print("--------------------------------------------------------------------------------------")
	print("Loss: " + str(rnn_results.history['loss'][-1]))
	print("MAE: " + str(rnn_results.history['mae'][-1]))

	# Plot Prediction Graph
	dates = pandas.to_datetime(dates)
	predictions = pandas.DataFrame(rnn.predict(train_input), columns = ["Stock Value Predictions"])
	real = pandas.DataFrame(train_label, columns = ["Real Stock Values"])
	
	final_dataframe = pandas.concat([dates, real, predictions], axis=1)
	final_dataframe = final_dataframe.sort_values(by = "Date")
	plot = final_dataframe.plot(x = "Date", y = ["Stock Value Predictions", "Real Stock Values"])
	plot.set_ylabel("Stock Value")
	pyplot.show()


	# *************************************************
	#		3. Save your model
	# *************************************************
	if (No_rel_days == 3): 
		rnn.model.save("./models/20986041_RNN_model.h5")
	elif (No_rel_days == 6):
		rnn.model.save("./models/6days_20986041_RNN_model.h5")
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------









