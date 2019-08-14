### Used to predict behavior and reward from trial factors 
import os
import keras
import numpy as np 
import pandas as pd

from keras_tqdm import TQDMCallback
from functions import settings as sett 
paths = sett.paths()

import wandb
from wandb.keras import WandbCallback


def load_data():
	factors = np.load(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '621510140', 'factors.npy'))
	meta_df = pd.read_csv(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '621510140', 'meta_df.csv'))

	return factors, meta_df

def train_test_folds(X, y):

	data = []
	for i in range(5):
		idx = [i for i in range(X.shape[0])]
		idx_to_remove = np.array([int(x+4*i) for x in [0, 1, 2, 3]])
		idx = np.array([int(i) for i in idx if i not in idx_to_remove])
		X_train = np.array(X[idx, :, :])
		X_test = np.array(X[idx_to_remove, :, :])
		y_train = np.array(y[idx, :])
		y_test = np.array(y[idx_to_remove, :])
		data.append((X_train, X_test, y_train, y_test))

	return data

def train_model(X_train, y_train):
	wandb.init(project='Predict_Trial')

	config = wandb.config
	config.layer_1_size  = 32
	config.layer_2_size = 32
	config.layer_3_size = 16
	config.dropout = 0.25
	config.epochs = 50

	inp = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
	x = keras.layers.LSTM(units=32, dropout=0.25,
						  input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)(inp)
	x = keras.layers.LSTM(units=16, dropout=0.25,
						  input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)(x)
	x = keras.layers.Dense(units=8)(x)
	output1 = keras.layers.Dense(1, activation='sigmoid')(x)
	output2 = keras.layers.Dense(1, activation='sigmoid')(x)
	output3 = keras.layers.Dense(1, activation='sigmoid')(x)
	output4 = keras.layers.Dense(1, activation='sigmoid')(x)

	model = keras.models.Model(inp, [output1, output2, output3, output4])
	model.compile(optimizer='rmsprop',
	              loss=['binary_crossentropy']*4,
	              metrics=['mae'])

	model.fit(X_train, y_train, epochs=1000, batch_size=2, verbose=0, callbacks=[WandbCallback()])
	# model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1)


	return model


def preprocess(factors, meta_df, trials):
	X = np.expand_dims(factors[2][trials, :], axis=1)
	new_X = np.empty(shape=(X.shape[0], X.shape[1], X.shape[2] + 1))
	
	# for i, tr in enumerate(X):
	# 	for j, dim in enumerate(tr):
	# 		if (meta_df.loc[[j], ['Odor Color']].values[0])[0] == 'red':
	# 			new_X[i][j] = np.append(X[i][j], 1)
	# 		elif (meta_df.loc[[j], ['Odor Color']].values[0])[0] == 'black':
	# 			new_X[i][j] = np.append(X[i][j], 0)
	# X = new_X

	new_X = np.empty(shape=(X.shape[0], X.shape[1], X.shape[2] + 1))
	for i, tr in enumerate(X):
		for j, dim in enumerate(tr):
			if (meta_df.loc[[j], ['Stimulus']].values[0])[0] == '+':
				new_X[i][j] = np.append(X[i][j], 1)
			elif (meta_df.loc[[j], ['Stimulus']].values[0])[0] == '-':
				new_X[i][j] = np.append(X[i][j], 0)
	X = new_X
	# One hot array: [HIT, MISS, FA, CR]
	y = np.zeros((X.shape[0], 4))
	for i, trial in enumerate(trials):
		if (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'HIT':
			y[i, 0] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'MISS':
			y[i, 1] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'FA':
			y[i, 2] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'CR':
			y[i, 3] = 1

	y = np.split(y, 4, axis=1)

	return X, y 

factors, meta_df = load_data()

days = {}
for day in set(meta_df['Day'].tolist()):
	days[day] = meta_df[meta_df['Day'] == day].index.tolist()


for day in days:
	X_train, y_train = preprocess(factors, meta_df, days[day])
	# y_test = np.split(y_test, 4, axis=1)

	model = train_model(X_train, y_train)

	block_performance = set(meta_df[meta_df['Block'] == block]['Performance'])

	prediction_behavior = model.predict(X)
	prediction_behavior = np.concatenate(prediction_behavior, axis=1)
	for i, row in enumerate(prediction_behavior):
		null_indices = [i for i in range(prediction_behavior.shape[1]) if i != np.argmax(row)]
		prediction_behavior[i, np.argmax(row)] = 1
		prediction_behavior[i, null_indices] = 0

	prediction_behavior = (np.sum(prediction_behavior[:,0]) + np.sum(prediction_behavior[:,3])) / X.shape[0]

	# scores.append((model.evaluate(X_test, y_test), block_performance, prediction_behavior))

	# ########### Printing Scores ###############
	# for i, b in enumerate(scores):
	# 	print('CV {}'.format(i))
	# 	print('HIT : %0.4f' % (b[0][5]*100))
	# 	print('MISS : %0.4f' % (b[0][6]*100))
	# 	print('FA : %0.4f' % (b[0][7]*100))
	# 	print('CR : %0.4f' % (b[0][8]*100))

	# print('Mean Overall Scores :')
	# print('Bloc {}'.format(block))
	# print('HIT : %0.4f' % (100*np.mean([scores[i][0][5] for i in range(len(scores))])))
	# print('MISS : %0.4f' % (100*np.mean([scores[i][0][6] for i in range(len(scores))])))
	# print('FA : %0.4f' % (100*np.mean([scores[i][0][7] for i in range(len(scores))])))
	# print('CR : %0.4f' % (100*np.mean([scores[i][0][8] for i in range(len(scores))])))
	# print('Model predicted performance : {} %'.format(100*np.mean([scores[i][2] for i in range(len(scores))])))
	# print('Mouse Performance : {} %'.format(list(scores[0][1])[0]))

def train_model_acti():
	acti = np.load(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '621510140', 'acti.npy'))
	factors = np.load(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '621510140', 'factors.npy'))
	meta_df = pd.read_csv(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '621510140', 'meta_df.csv'))

	# Separate trials into blocks
	blocks = {}
	for block in set(meta_df['Block'].tolist()):
		blocks[block] = meta_df[meta_df['Block'] == block].index.tolist()

	for block in blocks:
		X = factors[2][blocks[block], :]
		X = np.swapaxes(acti[:, :, blocks[block]], 0, 2)

		# One hot array: [HIT, MISS, FA, CR]
		y = np.zeros((X.shape[0], 4))
		for i, trial in enumerate(blocks):
			if (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'HIT':
				y[i, 0] = 1
			elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'MISS':
				y[i, 1] = 1
			elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'FA':
				y[i, 2] = 1
			elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'CR':
				y[i, 3] = 1

		# data = train_test_folds(X, y)
		scores = []
		# for (X_train, X_test, y_train, y_test) in data:
		y_train = np.split(y, 4, axis=1)
		X_train = X
		# y_test = np.split(y_test, 4, axis=1)

		inp = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
		x = keras.layers.LSTM(units=32, dropout=0.4,
							  input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)(inp)
		x = keras.layers.LSTM(units=32, dropout=0.4,
							  input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)(x)
		x = keras.layers.Dense(units=16)(x)
		output1 = keras.layers.Dense(1, activation='sigmoid')(x)
		output2 = keras.layers.Dense(1, activation='sigmoid')(x)
		output3 = keras.layers.Dense(1, activation='sigmoid')(x)
		output4 = keras.layers.Dense(1, activation='sigmoid')(x)


		model = keras.models.Model(inp, [output1, output2, output3, output4])
		model.compile(optimizer='rmsprop',
		              loss=['binary_crossentropy']*4,
		              metrics=['mae'])

		model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1)
		block_performance = set(meta_df[meta_df['Block'] == block]['Performance'])

		prediction_behavior = model.predict(X)
		prediction_behavior = np.concatenate(prediction_behavior, axis=1)
		for i, row in enumerate(prediction_behavior):
			null_indices = [i for i in range(prediction_behavior.shape[1]) if i != np.argmax(row)]
			prediction_behavior[i, np.argmax(row)] = 1
			prediction_behavior[i, null_indices] = 0

		prediction_behavior = (np.sum(prediction_behavior[:,0]) + np.sum(prediction_behavior[:,3])) / X.shape[0]

		scores.append((model.evaluate(X_test, y_test), block_performance, prediction_behavior))

		########### Printing Scores ###############
		for i, b in enumerate(scores):
			print('CV {}'.format(i))
			print('HIT : %0.4f' % (b[0][5]*100))
			print('MISS : %0.4f' % (b[0][6]*100))
			print('FA : %0.4f' % (b[0][7]*100))
			print('CR : %0.4f' % (b[0][8]*100))

		print('Mean Overall Scores :')
		print('Bloc {}'.format(block))
		print('HIT : %0.4f' % (100*np.mean([scores[i][0][5] for i in range(len(scores))])))
		print('MISS : %0.4f' % (100*np.mean([scores[i][0][6] for i in range(len(scores))])))
		print('FA : %0.4f' % (100*np.mean([scores[i][0][7] for i in range(len(scores))])))
		print('CR : %0.4f' % (100*np.mean([scores[i][0][8] for i in range(len(scores))])))
		print('Model predicted performance : {} %'.format(100*np.mean([scores[i][2] for i in range(len(scores))])))
		print('Mouse Performance : {} %'.format(list(scores[0][1])[0]))

train_model()
