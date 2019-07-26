import os
import numpy as np 
import pandas as pd
import keras

from functions import settings as sett 

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

paths = sett.paths()


acti = np.load(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '3834623592', 'acti.npy'))
meta_df = pd.read_csv(os.path.join(paths.path2Output, '020618-10', 'non_negative_parafac', 'random', '6', '3834623592', 'meta_df.csv'))


# Separate trials into blocks
blocks = {}
for block in set(meta_df['Block'].tolist()):
	blocks[block] = meta_df[meta_df['Block'] == block].index.tolist()

for block in blocks:
	X = np.swapaxes(acti[:, :, blocks[block]], 0, 2)

	# One hot array: [HIT, MISS, FA, CR]
	y = np.zeros((X.shape[0], 4))
	for i, trial in enumerate(blocks[block]):
		if (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'HIT':
			y[i, 0] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'MISS':
			y[i, 1] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'FA':
			y[i, 2] = 1
		elif (meta_df.loc[[trial], ['Behavior']].values[0])[0] == 'CR':
			y[i, 3] = 1
	data = train_test_folds(X, y)
	scores = []
	for (X_train, X_test, y_train, y_test) in data:
		y_train = np.split(y_train, 4, axis=1)
		y_test = np.split(y_test, 4, axis=1)

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

		model.fit(X_train, y_train, epochs=25, batch_size=4, verbose=1)
		block_performance = set(meta_df[meta_df['Block'] == block]['Performance'])
		
		prediction_behavior = model.predict(X)
		prediction_behavior = np.concatenate(prediction_behavior, axis=1)
		for i, row in enumerate(prediction_behavior):
			null_indices = [i for i in range(prediction_behavior.shape[1]) if i != np.argmax(row)]
			prediction_behavior[i, np.argmax(row)] = 1
			prediction_behavior[i, null_indices] = 0
		
		prediction_behavior = np.sum(prediction_behavior[:,0]) + np.sum(prediction_behavior[:,3]) / X.shape[0]

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

