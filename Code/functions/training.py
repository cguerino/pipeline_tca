import os
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
import tensorly as tl 
import numpy as np
from functions import tca_utils as tca
import tensorflow as tf
tl.set_backend('pytorch')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from functions import settings as sett 
paths = sett.paths()


def rf_cross_val_score(factors, meta_df, nb_estim):
	X, y_odor, y_rew = shuffle(factors[2], meta_df['Odor'].tolist(), meta_df['Reward'].tolist())
	clf = RandomForestClassifier(n_estimators=nb_estim, max_depth=None, min_samples_split=2, max_features='sqrt')
	X = StandardScaler().fit_transform(X)
	score_odor = cross_val_score(clf, X, y_odor, cv=8)
	score_rew = cross_val_score(clf, X, y_rew, cv=8)

	return score_odor, score_rew


def rf_oob_score(factors, meta_df, nb_estim):
	X, y_odor, y_rew = shuffle(factors[2], meta_df['Odor'].tolist(), meta_df['Reward'].tolist())
	print(X.shape)
	clf_odor = RandomForestClassifier(n_estimators=nb_estim, max_depth=None, min_samples_split=2, max_features='sqrt', oob_score=True)
	clf_rew = RandomForestClassifier(n_estimators=nb_estim, max_depth=None, min_samples_split=2, max_features='sqrt', oob_score=True)
	X = StandardScaler().fit_transform(X)
	clf_odor.fit(X, y_odor)
	score_odor = clf_odor.oob_score_
	clf_rew.fit(X, y_rew)
	score_rew = clf_rew.oob_score_

	return score_odor, score_rew, clf_odor, clf_rew

def rf_error_difference(factors, meta_df, animal, arguments, name):
	val_scores, oob_scores = [], []
	for nb_estim in tqdm(range(30, 200)):
		score_odor, score_rew = train.rf_cross_val_score(factors, meta_df, nb_estim)
		val_scores.append([score_odor.mean(), score_rew.mean()])
		score_odor, score_rew = train.rf_oob_score(factors, meta_df, nb_estim)
		oob_scores.append([score_odor, score_rew])

	x = [1 - val_scores[i][0] for i in range(len(val_scores))]
	y = [1 - oob_scores[i][0] for i in range(len(oob_scores))]

	plt.close()
	plt.plot(x, 'b')
	plt.plot(y, 'g')

	path = os.path.join(paths.path2Figures, animal, str(arguments['Function']), 
						str(arguments['Init']), str(arguments['Rank']), name)
	try:
	    os.makedirs(path)
	except:
	    FileExistsError
	
	plt.savefig(os.path.join(path, 'odor.png'))


	x = [1 - val_scores[i][1] for i in range(len(val_scores))]
	y = [1 - oob_scores[i][1] for i in range(len(oob_scores))]

	plt.close()
	plt.plot(x, 'b')
	plt.plot(y, 'g')
	plt.savefig(os.path.join(path, 'reward.png'))

def oob_error_rank(norm_acti, meta_df, init, animal, arguments, name):
	scores_odor, scores_rew = [], []
	plt.style.use('fivethirtyeight')
	for i in range(1, 11):
		factors, rec_errors = tl.decomposition.non_negative_parafac(norm_acti, i, n_iter_max=10000, tol=1e-07, 
																	verbose=1, return_errors=True, init=init)
		
		factors = [f.cpu().numpy() for f in factors]
		score_odor, score_rew = rf_oob_score(factors, meta_df, 50)
		scores_odor.append(score_odor)
		scores_rew.append(score_rew)
	plt.plot(np.arange(1, 11), scores_odor, 'r')
	plt.plot(np.arange(1, 11), scores_rew, 'b')
	plt.title('OOB_Score depending on rank')

	path = os.path.join(paths.path2Figures, animal, str(arguments['Function']), 
						str(arguments['Init']), name)
	try:
	    os.makedirs(path)
	except:
	    FileExistsError
	
	plt.savefig(os.path.join(path, 'oob_error_rank.png'))
	print('Done')


def feature_importance_roi_maps(factors, roi_tensor, clf, animal, arguments, name):
	feature_importances = clf.feature_importances_ 
	path = os.path.join(paths.path2Figures, animal, str(arguments['Function']), 
						str(arguments['Init']), str(arguments['Rank']), name)
	try:
	    os.makedirs(path)
	except:
	    FileExistsError

	for r in range(factors[0].shape[1]):
		plt.close()
		roi_map = tca.make_map(roi_tensor, factors[0][:, r])
		if np.min(factors[0][:, r]) < 0:
			plt.imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='coolwarm')
		else:
			plt.imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='hot')

		plt.title('Importance : {}'.format(feature_importances[r]))
		plt.savefig(os.path.join(path, 'Importance_map_{}.png'.format(feature_importances[r])))

def feature_importance_time_factor(factors, roi_tensor, clf, animal, arguments, name):
	feature_importances = clf.feature_importances_ 
	path = os.path.join(paths.path2Figures, animal, str(arguments['Function']), 
						str(arguments['Init']), str(arguments['Rank']), name)

	T = factors[1].shape[0]
	shaded = [7, 9]

	try:
	    os.makedirs(path)
	except:
	    FileExistsError

	for r in range(factors[1].shape[1]):
		plt.close()
		## plot time factors as a lineplot
		plt.plot(np.arange(1, T+1), factors[1][:, r], color='k', linewidth=2)
		# arrange labels on x axis
		plt.locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
		# color the shaded region
		plt.fill_betweenx([np.min(factors[1]), np.max(factors[1])+.01], 15*shaded[0],
								  15*shaded[1], facecolor='red', alpha=0.5)

		plt.title('Importance : {}'.format(feature_importances[r]))
		plt.savefig(os.path.join(path, 'Importance_time_{}.png'.format(feature_importances[r])))


def speed_tca(X):
	data_provider = Provider()
	data_provider.full_tensor = lambda: X
	env = Environment(data_provider, summary_path='/tmp/cp_' + '20')
	tf.compat.v1.disable_eager_execution()

	cp = CP_ALS(env)
	args = CP_ALS.CP_Args(rank=6, validation_internal=1)
	# build CP model with arguments
	cp.build_model(args)
	# train CP model with the maximum iteration of 100
	cp.train(100)
	# obtain factor matrices from trained model
	factor_matrices = cp.factors
	# obtain scaling vector from trained model
	lambdas = cp.lambdas

	return factors