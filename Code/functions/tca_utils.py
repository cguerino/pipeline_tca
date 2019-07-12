#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:10:47 2018

@author: Corentin Guérinot

Some useful functions for TCA
"""
import os 
import itertools
import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, parafac
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy as sci
import seaborn as sns
import statannot as sta


from functions import settings as sett

paths = sett.paths()


def raster_plot(tensor):
	"""Plot a raster activity plot of 5 trials chosen randomly in an activity tensor
	  
	Arguments:
		tensor {3d array} -- 3-dimensional array of shape (ROI, time, trial)
	"""

	fig = plt.figure(figsize=(30, 15))
	rand = np.random.randint(0, tensor.shape[2], 5)

	for i in range(5):
		
		fig.add_subplot(1, 5, i+1)
		plt.imshow(tensor[:, :, rand[i]], cmap='coolwarm', aspect='auto')
		
	plt.colorbar()
	
	
def make_map(roi_tensor, neuron_factor):
	"""Compute an image of the field of view with ROIs having different intensities
		
	Arguments:
		roi_tensor {boolean 3d array} -- 3-dimensional array of shape (512, 512, N) 
		where the slice (:,:,n) is a boolean mask for ROI n
		neuron_factor {list} -- list of length N with neuron factors of a component
		extracted from TCA
	
	Returns:
		2d array - image -- (512, 512) array
	"""
	
	roi_map = np.zeros([512, 512])
	for n in range(roi_tensor.shape[2]):
		roi_map += neuron_factor[n] * roi_tensor[:, :, n]
		
	return roi_map


def normalize(v):
	"""Normalize a vector with norm L2
	
	Arguments:
		v {array} -- any array
	
	Returns:
		array -- normalized array
	"""
	
	norm = np.linalg.norm(v)
	if norm == 0: 
		return v
	return v / norm


def rec_err(true_tensor, pred_tensor):
	"""Compute normalized distance between 2 arrays using L-2 norm
	
	Arguments:
		true_tensor {array} -- reference array (to which the normalization is done)
		pred_tensor {array} -- array to compare
	
	Returns:
		scalar -- normalized L-2 distance
	"""

	err = norm(true_tensor - pred_tensor) / norm(true_tensor)
	return err


def norm(tensor, order=2, axis=None):
	"""Computes the l-`order` norm of tensor

	Parameters
	----------
	tensor : ndarray
	order : int
	axis : int or tuple

	Returns
	-------
	float or tensor
		If `axis` is provided returns a tensor.
	"""
	# handle difference in default axis notation
	if axis == ():
		axis = None

	if order == 'inf':
		return np.max(np.abs(tensor), axis=axis)
	if order == 1:
		return np.sum(np.abs(tensor), axis=axis)
	elif order == 2:
		return np.sqrt(np.sum(tensor**2, axis=axis))
	else:
		return np.sum(np.abs(tensor)**order, axis=axis)**(1/order)


def norm_factors(factors):
	"""Normalize the factors output of TCA
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
	
	Returns:
		list -- list of 3 arrays normalized per column 
	"""
	
	norm_list = []
	
	for factor in factors:
		
		rank = factor.shape[1]
		norm_factor = np.zeros_like(factor)
		
		for i in range(rank):
			norm_factor[:, i] = normalize(factor[:, i])
			
		norm_list.append(norm_factor)
	
	return norm_list


def norm_tensor(acti):
	"""Normalize activity tensor to a range (0, 1)
	
	Normalization consisting in adding 1 to all value to make all entries positive, 
	and then divide each trial activity by its maximum activity 
	
	Arguments:
		acti {3d array} -- activity array of shape (ROI, time, trial)
	
	Returns:
		3d array -- normalized activity array
	"""
	
	N, T, _ = acti.shape
	norm_acti = np.zeros_like(acti)
	max_amp = np.nanmax(acti, axis=(0, 1))

	for t in range(T):
		for n in range(N):
			norm_acti[n, t, :] = np.divide(acti[n, t, :] + 1, max_amp + 1)

	return norm_acti


def day_limits(day, drop_trial):
	"""Return the trial reference for the change in day - taking 
	the dropped trials into account
	
	Arguments:
		day {list} -- day reference for each trial
		drop_trial {list} -- reference of trials removed in pre-processing
	
	Returns:
		list -- reference of first trial of each day of experiment
	"""

	limit_day = []
	
	day_drop = np.delete(day, drop_trial)
	tic = 1
	
	for i, _ in enumerate(day_drop):
		
		if day_drop[i] > tic:
			limit_day.append(i)
			tic += 1
		
	return limit_day


def make_score(learning, beh):
	"""Compute learning score for each block
	
	Arguments:
		learning {list} -- learning state code for trials
		(Non-learning: 0, Learning: 1)
		beh {type} -- behavioral code for trials
		(Hit: 1, Miss: 2, CR: 3, FA: 4)
	
	Returns:
		tuple -- number of blocks, list of learning scores for each block
	"""
	
	n_blocks = len(beh) // 20
	idx = 0
	learn_score = []

	for _ in range(n_blocks):

		score = 0
		trial = 0

		while trial < 20:

			if learning[idx] > 0:

				if beh[idx] == 1:
					score += 5
				if beh[idx] == 3:
					score += 5

			trial += 1
			idx += 1

		learn_score.append(score)
	
	return n_blocks, learn_score


def get_limits(n_blocks, drop_trial):
	"""Compute trials boundaries for each block - 
	taking the dropped trials into account
	
	Arguments:
		n_blocks {scalar} -- number of blocks
		drop_trial {list} -- reference of trials removed in pre-processing
	
	Returns:
		list -- list of 2-elements lists denoting the trials limits of each block
	"""

	limit_block = []
	offset_list = []
	right = 0

	for block in range(n_blocks):

		left = right
		right = left + 20
		local_offset = 0

		for trial in drop_trial:
			if trial >= block*20:
				if trial < (block + 1)*20:
					local_offset += 1

		right += - local_offset

		offset_list.append(local_offset)
		limit_block.append([left, right])
		
	return limit_block


def convert_color(learn_score):
	"""Convert learning score into color code for display
	
	Diverging colormap from red to green, grey for non-learning trials
	
	Arguments:
		learn_score {list} -- learning score for each block
	
	Returns:
		list -- color list coded for learning
	"""

	learn_color = []

	for score in learn_score:
		if score == 0:
			learn_color.append('grey')
		elif (score > 0) & (score <= 35):
			learn_color.append('maroon')
		elif (score > 35) & (score <= 50):
			learn_color.append('darkred')
		elif score == 55:
			learn_color.append('red')
		elif score == 60:
			learn_color.append('orangered')
		elif score == 65:
			learn_color.append('darkorange')
		elif score == 70:
			learn_color.append('orange')
		elif score == 75:
			learn_color.append('goldenrod')
		elif score == 80:
			learn_color.append('gold')
		elif score == 85:
			learn_color.append('yellowgreen')
		elif score == 90:
			learn_color.append('lawngreen')
		elif score == 95:
			learn_color.append('lime')
		elif score == 100:
			learn_color.append('limegreen')
  
	return learn_color


def behaviorgram(beh_mat, drop_trial):
	"""Convenient method for plotting, yielding learning color for each block,
	the limit trials for each block and the reference for the first trial of each day 
	
	Arguments:
		beh_mat {array} -- 4-columns array [odor, behavior, learning, day]
		drop_trial {list} -- reference of trials removed in pre-processing
	
	Returns:
		tuple -- color list coded for learning, list of each block boundaries,
				 list of each first day trial
	"""
	
	beh = beh_mat[:, 1]
	learning = beh_mat[:, 2]
	day = beh_mat[:, 3]
	
	n_blocks, learn_score = make_score(learning, beh)
	
	limit_day = day_limits(day, drop_trial)
	limit_block = get_limits(n_blocks, drop_trial)
	learn_color = convert_color(learn_score)
	
	return learn_color, limit_block, limit_day


def give_order(factors):
	"""Compute order of temporal factors based on activity onset 
	
	For each component, we take the first timeframe where activity exceeds
	twice the standard deviation. Then we order them in ascending order
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
	
	Returns:
		list -- permutation of length the rank of associated TCA
	"""

	rank = factors[0].shape[1]
	onset = []

	for r in range(1, rank):
		comp = factors[1][:, r]
		thres = 2 * np.std(comp) + comp[0]
		t = 0
		while comp[t] < thres:
			if t < 284:
				t += 1
			else:
				break
		onset.append(t)

	order = []
	ti = np.copy(onset)
	onset.sort()

	for t in onset:
		i = 0
		while t != ti[i]:
			i += 1
		order.append(i)
	
	return order


def ord_fact(factors, order):
	"""Permute the component in each TCA factors array
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
		order {list} -- permutation of length the rank of TCA

	Returns:
		list -- list of 3 arrays whose components are ordered by activity onset
	"""

	ord_factors = []

	for factor in factors:
		ord_factor = np.zeros_like(factor)
		ord_factor[:, 0] = factor[:, 0]
		for i, o in enumerate(order):
			ord_factor[:, i+1] = factor[:, o+1]
		ord_factors.append(ord_factor)
		
	return ord_factors


def reco(acti, factors, rank, nidx=None, kidx=None):
	"""Show reconstruction of activity trace, true vs predicted
	
	 
	Arguments:
		acti {array} -- 3-dimensional activity array
		factors {list} -- list of 3 arrays containing the TCA factors
		rank {scalar} -- component from which reconstructed trace is most ressembling
	
	Keyword Arguments:
		nidx {scalar} -- ROI whose trace will be reconstructed (default: {None})
							if None, nidx chosen to maximize neuron factors of given rank
		kidx {scalar} -- Trial whose trace will be reconstructed (default: {None})
							if None, kidx chosen to maximize trial factors of given rank
	"""
	
	N, T, K = acti.shape
	norm_acti = norm_tensor(acti)
	
	pred_tensor = tl.kruskal_to_tensor(factors)
	factors = ord_fact(factors, give_order(factors))
	
	if nidx is None:
		_, nidx = max((factors[0][:, rank][i], i) for i in range(N))
	if kidx is None:
		_, kidx = max((factors[2][:, rank][i], i) for i in range(K))
	 
	plt.plot(pred_tensor[nidx, :, kidx], color='orangered', linewidth=2, label='Model')
	plt.plot(norm_acti[nidx, :, kidx], color='blue', linewidth=1, label='True')
	plt.xlabel('Time (s)', {'fontsize': 'medium', 'fontweight' : 'bold'})
	plt.ylabel('Normalized df/f0', {'fontsize': 'medium', 'fontweight' : 'bold'})
	plt.locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
	plt.fill_betweenx([0, 1], 105, 135, facecolor='red', alpha=0.3, label='Odor Pres.')

	time_index = list(np.arange(0, T//15 + 1, 2))
	plt.xticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270], time_index)
	plt.legend(loc=1)
	
	r2 = round(r2_score(pred_tensor[nidx, :, kidx], norm_acti[nidx, :, kidx]), 3)
	plt.text(x=0, y=0.9, s='R2 = {0}'.format(r2))
	
#    plt.show()
	
	
def make_box_df(factors, sel, b, color_df):
	"""Build a dataframe convenient for a boxplot of trial factors
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
		sel {scalar} -- component selected
		b {tuple} -- color list coded for learning, list of each block boundaries,
				 list of each first day trial
		color_df {pandas dataframe} -- columns [Odor, Reward, Day, Behavior] with color coded
	
	Returns:
		pandas dataframe -- columns [Odor, Reward, Day, Behavior, Block, Trial Factor]
	"""
	
	lims = b[1]
	block_list = []
	for i, lim in enumerate(lims):
		a, b = lim
		for _ in range(a, b):
			block_list.append(i)

	order = give_order(factors)
	factors = ord_fact(factors, order)

	box_df = pd.DataFrame(index=color_df.index, columns=['Odor', 'Block', 'Trial Factor'])

	box_df['Odor'] = color_df['Odor']
	box_df.replace('red', 'S +', inplace=True)
	box_df.replace('black', 'S -', inplace=True)

	box_df['Reward'] = color_df['Reward']
	box_df.replace('purple', 'Reward', inplace=True)
	box_df.replace('grey', 'No Reward', inplace=True)

	box_df['Day'] = color_df['Day']
	for i, day in enumerate(color_df['Day'].unique()):
		box_df.replace(day, i, inplace=True)

	box_df['Behavior'] = color_df['Behavior']
	box_df.replace('black', 'Miss', inplace=True)
	box_df.replace('blue', 'CR', inplace=True)
	box_df.replace('yellow', 'FA', inplace=True)
	box_df.replace('red', 'Hit', inplace=True)

	box_df['Block'] = pd.Series(block_list)

	box_df['Trial Factor'] = pd.Series(factors[2][:, sel])
	
	return box_df


def box_zoom(factors, sel, hue, b, color_df, palette=None, stat=False):
	"""Display boxplot for a given TCA-component associated trial factor
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
		sel {scalar} -- component selected
		hue {string} -- entry in the color_df to color code trial factors
		b {tuple} -- color list coded for learning, list of each block boundaries,
				 list of each first day trial
		color_df {pandas dataframe} -- columns [Odor, Reward, Day, Behavior] with color coded
	
	Keyword Arguments:
		palette {list} -- color palette for plotting (default: {['red', 'black']})
		stat {bool} -- add significance test stars (default: {False})
	"""

	box_df = make_box_df(factors, sel, b, color_df)
	n_blocks = len(b[1])
	if palette is None:
		palette = ['red', 'black']
	
	plt.rcParams['figure.figsize'] = 14, 6
	#fig = plt.figure(figsize=(14, 6))

	ax = sns.boxplot(x="Block", y="Trial Factor", hue=hue, data=box_df,
					 palette=palette, dodge=False,
					 linewidth=2, fliersize=2, width=.3)
	xmin, xmax, ymin, ymax = ax.axis()

	if stat:
		c1, c2 = box_df[hue].unique()[:2]
		for i in range(n_blocks):
			if len(box_df[hue][box_df['Block'] == i].unique()) < 2:
				continue
			sta.add_stat_annotation(ax, data=box_df, x="Block", y="Trial Factor", hue=hue,
									boxPairList=[((i, c1), (i, c2))], test='t-test',
									textFormat='star', loc='inside', fontsize='large',
									lineYOffsetAxesCoord=0.05, linewidth=0, verbose=0)

	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax*1.15)

	plt.show()


def seq_parafac(input_tensor, max_rank, nb_trial, pred_df, tol=1e-07, mode='non-negative'):
	"""Sequential TCA for model selection
	
	This method computes TCA with a number of components ranging from 1 to the maximum rank indicated,
	and stores reconstruction error, similarity, sparsity for each model obtained. It fits also random
	forest classifiers to predict reward and odor for each trial using trial factors, and it stores
	the associated prediction accuracy.
	
	Arguments:
		input_tensor {array} -- 3-dimensional activity array
		max_rank {scalar} -- maximum rank for TCA
		nb_trial {scalar} -- number of replication of same TCA
		pred_df {pandas dataframe} -- reward and odor information for each trial
	
	Keyword Arguments:
		tol {scalar} -- tolerance for optimization convergence
		mode {string} -- version of TCA to compute, classic or non-negative (default: {'non-negative'})
	
	Returns:
		pandas err_df -- reconstruction error for each TCA run
		pandas sim_df -- similarity against best fit for each TCA run
		pandas spa_df -- sparsity for each TCA run
		pandas odor_df -- odor prediction accuracy for each TCA run
		pandas rew_df -- reward prediction accuracy for each TCA run
	"""

	# lists used for output dataframes
	rank_err = []
	rank_sim = []
	err_list = []
	sim_list = []
	spa_list = []
	
	odor_acc = []
	odor_std = []
	rew_acc = []
	rew_std = []
	

	for rank in np.arange(1, max_rank+1):

		# in this list we store factors extracted with TCA for each run
		pred_fac = []
		# minimal error initialized at maximum 1, useful to identify best fit model
		min_err = 1
		# index of the best fit model in the factors list
		min_idx = 0

		# we iterate over nb_trial, the number of replicates of TCA to run
		# it allows stability check, i.e by computing sparsity
		for trial in range(nb_trial):

			# verbose to know which replicate is running
			print('Trial', trial)

			# build a list useful for err_df
			rank_err.append(rank)
		
			if mode == 'non-negative':
				# where TCA is actually computed, here in its non-negative version
				pred_fac.append(non_negative_parafac(input_tensor, rank=rank, n_iter_max=1000,
													 init='svd', tol=tol, verbose=1))
			else:
				# where TCA is actually computed, in its classic version
				pred_fac.append(parafac(input_tensor, rank=rank, n_iter_max=5000,
										init='random', tol=tol))
			# we store all factors in a list, to be able to compute model similarity in the end

			# transform pred_fac from kruskal form (list of factors) to full-tensor form
			pred_tensor = tl.kruskal_to_tensor(pred_fac[trial])
			# compute reconstruction error, L2 distance from predicted to original tensor 
			err = rec_err(input_tensor, pred_tensor)
			err_list.append(err)
			
			# here we compute sparsity, the proportion of almost-zero elements 
			nb_nonzero = 0
			tot_size = 0
			
			for i in range(len(pred_fac[trial])):
				nb_nonzero += np.count_nonzero(np.round(pred_fac[trial][i], 2))
				tot_size += pred_fac[trial][i].size
			
			spa = 1 - nb_nonzero / tot_size
			spa_list.append(spa)
			
			# we shuffle samples matrix (here trial factors) and labels (odor and reward)
			# using same permutation
			X, y_odor, y_rew = shuffle(pred_fac[trial][2], pred_df['Odor'].tolist(),
									   pred_df['Reward'].tolist())

			# initialize random forest classifier
			clf = RandomForestClassifier(n_estimators=50, max_depth=None,
										 min_samples_split=2, max_features='sqrt')

			# scale the data before fitting
			X = StandardScaler().fit_transform(X)
			
			# compute cross validated prediction accuracy for odor and reward
			odor_acc.append(cross_val_score(clf, X, y_odor, cv=8).mean())
			odor_std.append(cross_val_score(clf, X, y_odor, cv=8).std())
			rew_acc.append(cross_val_score(clf, X, y_rew, cv=8).mean())
			rew_std.append(cross_val_score(clf, X, y_rew, cv=8).std())

			# we keep track of the model having lowest reconstruction error
			# we will use this model as a reference to compute model similarity
			if err < min_err:
				min_err = err
				min_idx = trial

		# we iterate again over all computed models to calculate similarity 
		# versus best fit model
		for trial in range(nb_trial):
			
			# if the model is the best fit, do nothing
			if trial == min_idx:
				continue
			
			# build a list useful for sim_df
			rank_sim.append(rank)

			# align factors to compute similarity
			sim_list.append(tt.kruskal_align(tt.tensors.KTensor(pred_fac[min_idx]), 
											 tt.tensors.KTensor(pred_fac[trial]), 
											 permute_U=True, permute_V=True))
			
	# build dataframes to store results
	err_df = pd.DataFrame(data=np.transpose([rank_err, err_list]), 
						  columns=['Rank', 'Reconstruction Error'])
	sim_df = pd.DataFrame(data=np.transpose([rank_sim, sim_list]), 
						  columns=['Rank', 'Similarity'])
	spa_df = pd.DataFrame(data=np.transpose([rank_err, spa_list]), 
						  columns=['Rank', 'Sparsity'])
	odor_df = pd.DataFrame(data=np.transpose([rank_err, odor_acc, odor_std]),
						   columns=['Rank', 'Accuracy - Odor Prediction', 'Std - Odor Prediction'])
	rew_df = pd.DataFrame(data=np.transpose([rank_err, rew_acc, rew_std]),
						  columns=['Rank', 'Accuracy - Reward Prediction',
								   'Std - Reward Prediction'])
	
	return err_df, sim_df, spa_df, odor_df, rew_df


def factorplot(factors, roi_tensor, meta_df, balance=True, color='k', shaded=None, order=False, path=None):
	"""Display the factors extracted with TCA
	
	The TCA extracted factors are represented in 3 columns and as many rows as there are components.
	On the first column neuron factors are represented on the ROI map.
	On the second column temporal factors are represented.
	On the third column trial factors are represented.
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
		roi_tensor {boolean 3d array} -- 3-dimensional array of shape (512, 512, N) 
	
	Keyword Arguments:
		b {tuple} -- behaviorgram: color list coded for learning, list of each block boundaries,
					 list of each first day trial (default: None)
		balance {bool} -- whether factors be normalized over modes and components (default: {True})
		color {list} -- color code for each trial factor (default: {'k'})
		shaded {list} -- interval in seconds to be shaded in temporal factors (default: {None})
		order {bool} -- whether components should be ordered by actity onset (default: {False})
		path {string} -- destination folder for saving (default: None)
	"""
	
	# factors is the list of 3 factor matrices - Kruskal form of tensor
	
	# whether or not factors columns should be normalized to unit norm
	if balance:
		# Compute norms along columns for each factor matrix
		# norms is a list of 3 arrays of length rank
		norms = [sci.linalg.norm(f, axis=0) for f in factors]

		# Multiply norms across all modes to have 1 norm per component
		# lam is a list of length rank
		lam = sci.multiply.reduce(norms) ** (1/3)

		# Update factors to normalize each columns to unit norm
		factors = [f * (lam / fn) for f, fn in zip(factors, norms)]
	
	# wheter or not components should be ordered by activity onset
	if order:
		factors = ord_fact(factors, give_order(factors))

	# rank is the number of components of TCA - as well as the number of columns in factor matrices
	rank = factors[0].shape[1]
	# T is the number of timeframes for each trial, usually 285
	T = factors[1].shape[0]
	# K is the number of trials
	K = factors[2].shape[0]


	limit_block = [[meta_df[meta_df['Block'] == i].index.tolist()[0], meta_df[meta_df['Block'] == i].index.tolist()[-1]] for i in set(meta_df['Block'].tolist())]
	limit_day = [meta_df[meta_df['Day'] == i].index.tolist()[-1] for i in set(meta_df['Day'].tolist())]
	learn_color = [meta_df['Performance Color'].iloc[x[0]] for x in limit_block]
	top = np.max(factors[2])

	# by default the shaded interval for trial factors
	# corresponds to the odor presentation
	if shaded is None:
		# interval in seconds, 1 second = 15 timeframes
		shaded = [7, 9]
	
	# initiate the plotting object
	fig, axarr = plt.subplots(rank, 3, sharex='col', figsize=(15, rank*3))

	# for each of the component r
	for r in range(rank):

		## plot neuron factors on the ROI map
		# generate the image with ROI binary tensor and neuron factors
		roi_map = make_map(roi_tensor, factors[0][:, r])
		# plot as an image, beware normalized colormap
		axarr[r, 0].imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='hot')

		## plot time factors as a lineplot
		axarr[r, 1].plot(np.arange(1, T+1), factors[1][:, r], color='k', linewidth=2)
		# arrange labels on x axis
		axarr[r, 1].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
		# color the shaded region
		axarr[r, 1].fill_betweenx([0, np.max(factors[1])+.01], 15*shaded[0],
								  15*shaded[1], facecolor='red', alpha=0.5)

		## plot trial factors as a scatter plot
		axarr[r, 2].scatter(np.arange(1, K+1), factors[2][:, r], c=color)
		# arrange labels on x axis
		axarr[r, 2].locator_params(nbins=K//20, steps=[1, 2, 5, 10], min_n_ticks=K//20)
		
		# add information from behaviorgram if needed

		# iterate over blocks
		for i, block in enumerate(limit_block):
			# color a region over trial factors corresponding to a given block
			# the color denotes the learning score for the given block
			axarr[r, 2].fill_betweenx([1.05 * top, 1.25 * top], block[0], block[1], 
									  facecolor=learn_color[i], alpha=1)
		# iterate over days
		for limit in limit_day:
			# plot a black line for day shift between learning score colors
			axarr[r, 2].axvline(limit, 0.75, 1, linewidth=2, color='black')
		
		# for mode 1 and 2 (i.e temporal and trial factors)
		for i in [1, 2]:

			# format axes, remove spines for all components
			axarr[r, i].spines['top'].set_visible(False)
			axarr[r, i].spines['right'].set_visible(False)

			# remove xticks on all but bottom row, to keep legend on this row
			if r != rank-1:
				plt.setp(axarr[r, i].get_xticklabels(), visible=False)
		
		# remove axes, spines and labels for neuron factors
		axarr[r, 0].tick_params(axis='both', which='both', bottom=False, top=False,
								labelbottom=False, right=False, left=False, labelleft=False)

	# set titles for top row and legend for bottom row
	axarr[0, 0].set_title('Neuron Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	axarr[0, 1].set_title('Temporal Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	axarr[0, 2].set_title('Trial Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	
	# set label for bottom row trial factors
	axarr[rank-1, 0].set_xlabel('ROI map', {'fontsize': 'large', 'fontweight' : 'bold',
											'verticalalignment' : 'top'})

	# generate time index in seconds for temporal factors
	time_index = list(np.arange(0, T//15 + 1, 2))
	# insert another 0 to preserve length
	time_index.insert(0, 1)
	# set label for bottom row temporal factors
	axarr[rank-1, 1].set_xlabel('Time (s)', {'fontsize': 'large', 'fontweight' : 'bold'})
	# set ticks labels for bottom row temporal factors
	axarr[rank-1, 1].set_xticklabels(time_index)
	
	# generate trial index for trial factors
	trial_index = list(np.arange(0, K//20 + 2))
	# insert another 0 to preserve length
	trial_index.insert(0, 1)
	# set label for bottom row trial factors
	axarr[rank-1, 2].set_xlabel('Block', {'fontsize': 'large', 'fontweight' : 'bold'})
	# set ticks labels for bottom trial factors 
	axarr[rank-1, 2].set_xticklabels(trial_index)
	
	## link y-axes within columns
	# iterate over modes
	for i in range(3):
		# get amplitudes for each component
		yl = [a.get_ylim() for a in axarr[:, i]]
		# get maximum amplitudes, global minimum and maximum for factors
		y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
		# set same plotting intervals across components 
		_ = [a.set_ylim((y0, y1)) for a in axarr[:, i]]

	## format y-ticks
	# iterate over components
	for r in range(rank):
		# iterate over modes
		for i in range(3):
			# limit to two labels, minimum and maximum for y axis
			axarr[r, i].set_ylim(np.round(axarr[r, i].get_ylim(), 2))
			# set ticks accordingly
			axarr[r, i].set_yticks([0, np.round(axarr[r, i].get_ylim(), 2)[1]])

	# make so that plots are tightly presented
	plt.tight_layout()

	# if a path is given, save the obtained figure
	if path is not None:
		plt.savefig(path)

	# display figure
	plt.savefig(os.path.join(paths.path2Figures, 'factorplot.png'))

	plt.show(fig)
	

def factorplot_singlecomp(factors, roi_tensor, b=None, balance=True, color='k', shaded=None):
	"""Display the factors extracted with TCA, special case for 1 component

	- see factorplot
	
	The TCA extracted factors are represented in 3 columns and as many rows as there are components.
	On the first column neuron factors are represented on the ROI map.
	On the second column temporal factors are represented.
	On the third column trial factors are represented.
	
	Arguments:
		factors {list} -- list of 3 arrays containing the TCA factors
		roi_tensor {boolean 3d array} -- 3-dimensional array of shape (512, 512, N) 
	
	Keyword Arguments:
		b {tuple} -- behaviorgram: color list coded for learning, list of each block boundaries,
					 list of each first day trial (default: None)
		balance {bool} -- whether factors be normalized over modes and components (default: {True})
		color {list} -- color code for each trial factor (default: {'k'})
		shaded {list} -- interval in seconds to be shaded in temporal factors (default: {None})
		order {bool} -- whether components should be ordered by actity onset (default: {False})
		path {string} -- destination folder for saving (default: None)
	"""
	
	# factors is the list of 3 factor matrices - Kruskal form of tensor
	
	if balance:
		# Compute norms along columns for each factor matrix
		norms = [sci.linalg.norm(f, axis=0) for f in factors]

		# Multiply norms across all modes
		lam = sci.multiply.reduce(norms) ** (1/3)

		# Update factors
		factors = [f * (lam / fn) for f, fn in zip(factors, norms)]

	T = factors[1].shape[0]
	K = factors[2].shape[0]

	#behaviorgram
	if b is not None:
		learn_color, limit_block, limit_day = b
		top = np.max(factors[2])

	if shaded is None:
		shaded = [7, 9]
	
	# initiate the plotting object
	_, axarr = plt.subplots(1, 3, sharex='col', figsize=(15, 3))

	# plot neuron factors on the ROI map
	roi_map = make_map(roi_tensor, factors[0])
	axarr[0].imshow(roi_map, vmin=0, vmax=np.max(factors[0]), cmap='hot')
	
	# plot time factors as a lineplot
	axarr[1].plot(np.arange(1, T+1), factors[1], color='k', linewidth=2)
	axarr[1].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
	axarr[1].fill_betweenx([0, np.max(factors[1])+.01], 15*shaded[0],
						   15*shaded[1], facecolor='red', alpha=0.5)
	# plot trial factors as a scatter plot
	axarr[2].scatter(np.arange(1, K+1), factors[2], c=color)
	axarr[2].locator_params(nbins=K//20, steps=[1, 2, 5, 10], min_n_ticks=K//20)
	
	if b is not None:
		for i, block in enumerate(limit_block):
			axarr[2].fill_betweenx([1.05 * top, 1.25 * top], block[0], block[1], 
								   facecolor=learn_color[i], alpha=1)
		
		for limit in limit_day:
			axarr[2].axvline(limit, 0.75, 1, linewidth=2, color='black')
	
	for i in [1, 2]:

		# format axes
		axarr[i].spines['top'].set_visible(False)
		axarr[i].spines['right'].set_visible(False)
	
	axarr[0].tick_params(axis='both', which='both', bottom=False, top=False,
						 labelbottom=False, right=False, left=False, labelleft=False)

	# set titles for top row and legend for bottom row
	axarr[0].set_title('Neuron Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	axarr[1].set_title('Temporal Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	axarr[2].set_title('Trial Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
	
	axarr[0].set_xlabel('ROI map',
						{'fontsize': 'large', 'fontweight' : 'bold', 'verticalalignment' : 'top'})

	time_index = list(np.arange(0, T//15 + 1, 2))
	time_index.insert(0, 1)
	axarr[1].set_xlabel('Time (s)', {'fontsize': 'large', 'fontweight' : 'bold'})
	axarr[1].set_xticklabels(time_index)
	
	trial_index = list(np.arange(0, K//20 + 2))
	trial_index.insert(0, 1)
	axarr[2].set_xlabel('Block', {'fontsize': 'large', 'fontweight' : 'bold'})
	axarr[2].set_xticklabels(trial_index)
	
	_, ymax1 = np.round(axarr[1].get_ylim(), 2)
	axarr[1].set_ylim((0, ymax1))
	axarr[1].set_yticks([0, ymax1])
	_, ymax2 = np.round(axarr[2].get_ylim(), 2)
	axarr[2].set_ylim((0, ymax2))
	axarr[2].set_yticks([0, ymax2])

	# make so that plots are tightly presented
	plt.tight_layout()
	plt.savefig(os.path.join(paths.path2Figures, 'factorplot.png'))
	plt.show()


def custom_initialize_factors(tensor, rank, init='svd', svd='numpy_svd', random_state=None, non_negative=False):
    r"""Initialize factors used in `parafac`.

    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
    init : {'svd', 'random'}, optional
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned

    Returns
    -------
    factors : ndarray list
        List of initialized factors of the CP decomposition where element `i`
        is of shape (tensor.shape[i], rank)

    """
    rng = tl.random.check_random_state(random_state)

    if init == 'random':
        factors = [tl.tensor(rng.random_sample((tensor.shape[i], rank)), **tl.context(tensor)) for i in range(tl.ndim(tensor))]
        custom_factors = [f if i == 0 else tl.abs(f) for i, f in enumerate(factors)]

    elif init == 'svd':
        try:
            svd_fun = tl.SVD_FUNS[svd]
        except KeyError:
            message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
                    svd, tl.get_backend(), tl.SVD_FUNS)
            raise ValueError(message)

        factors = []
        for mode in range(tl.ndim(tensor)):
            U, _, _ = svd_fun(tl.base.unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                # TODO: this is a hack but it seems to do the job for now
                # factor = tl.tensor(np.zeros((U.shape[0], rank)), **tl.context(tensor))
                # factor[:, tensor.shape[mode]:] = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                # factor[:, :tensor.shape[mode]] = U
                random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
                U = tl.concatenate([U, random_part], axis=1)
            
            if mode == 0:
                factors.append(U[:, :rank])
            else:
                 factors.append(tl.abs(U[:, :rank]))
        return factors

    raise ValueError('Initialization method "{}" not recognized'.format(init))

def custom_parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd', tol=1e-8,
            orthogonalise=False, random_state=None, verbose=False, return_errors=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

    Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

        ``tensor = [| factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random'}, optional
        Type of factor matrix initialization. See `initialize_factors`.
    svd : str, default is 'numpy_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors


    Returns
    -------
    factors : ndarray list
        List of factors of the CP decomposition element `i` is of shape
        (tensor.shape[i], rank)
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] tl.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    # if orthogonalise and not isinstance(orthogonalise, int):
    #     orthogonalise = n_iter_max

    factors = custom_initialize_factors(tensor, rank, init=init, svd=svd, random_state=random_state)
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    epsilon = 10e-12

    for iteration in range(n_iter_max):
        # if orthogonalise and iteration <= orthogonalise:
        #     factor = [tl.qr(factor)[0] for factor in factors]

        for mode in range(tl.ndim(tensor)):
            if mode == 0:
                pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
                for i, factor in enumerate(factors):
                    if i != mode:
                        pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
               
                factor = tl.dot(tl.base.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
                factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
                factors[mode] = factor
            else:
                sub_indices = [i for i, j in enumerate(factors) if i != mode]
                for i, e in enumerate(sub_indices):
                    if i:
                        accum = accum*tl.dot(tl.transpose(factors[e]), factors[e])
                    else:
                        accum = tl.dot(tl.transpose(factors[e]), factors[e])

                numerator = tl.dot(tl.base.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
                numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
                denominator = tl.dot(factors[mode], accum)
                denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
                factors[mode] = factors[mode]* numerator / denominator

        if tol:
            rec_error = tl.norm(tensor - tl.kruskal_tensor.kruskal_to_tensor(factors), 2) / norm_tensor
            rec_errors.append(rec_error)

            if iteration > 1:
                if verbose:
                    print('reconstruction error={}, variation={}.'.format(
                        rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

                if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                    if verbose:
                        print('converged in {} iterations.'.format(iteration))
                    break
    
    np.save(os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Output', 'rank{0}_factors'.format(rank)), factors)
    np.save(os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Output', 'rank{0}_errors'.format(rank)), rec_errors)

    if return_errors:
        return factors, rec_errors
    else:
        return factors
