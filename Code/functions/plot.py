import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from functions import tca_utils as tca
from functions import settings as sett

paths = sett.paths()


def map_of_rois(acti, roi_tensor):
	N, T, K = acti.shape
	plt.imshow(tca.make_map(roi_tensor, np.ones(roi_tensor.shape[2])), cmap='hot')

	fig = plt.figure(figsize=(12,12))

	fig.add_subplot(2,2,1)

	nan_per_trial = pd.Series(np.isnan(acti[0,:,:]).sum(axis=0))
	nan_per_trial.plot(kind = 'bar', color = 'blue', width = 1)
	plt.xticks(nan_per_trial.index[::40],  nan_per_trial.index[::40], rotation = 'vertical')
	plt.xlabel('Trials')
	plt.ylabel('Number of NaN')


	fig.add_subplot(2,2,2)

	nan_per_timeframe = pd.Series(np.isnan(acti[0,:,:]).sum(axis=1))
	nan_per_timeframe.plot(kind='bar', color='blue', width=1)
	plt.xticks(nan_per_trial.index[0:T:15], rotation='horizontal')
	plt.xlabel('Time')

	plt.fill_betweenx([0, 1.2 * np.max(nan_per_timeframe)], 105, 135, facecolor='red', alpha=0.5)

	plt.tight_layout()
	plt.savefig(os.path.join(paths.path2Figures, 'map_of_rois.png'))

def nan_ivestigation():
	# We remove the trials with only NaN
	nan_per_trial = pd.Series(np.isnan(acti[0,:,:]).sum(axis=0))
	drop_trial = nan_per_trial[nan_per_trial == 285].index.tolist()
	nan_per_trial.drop(drop_trial, inplace = True)

	# Investigate consecutive NaN
	df = pd.DataFrame(acti[0,:,:])
	df = df.fillna(-2)
	cons_nan_per_trial = [max(sum(1 for x in j if x == -2) for i, j in itertools.groupby(df[trial])) for trial in df]
	nb_drop = [sum(1 for trial in cons_nan_per_trial if trial > threshold) for threshold in range(30)]

	#Display some results    
	fig = plt.figure(figsize=(10,10))

	fig.add_subplot(2,2,1)
	plt.bar(range(30), nb_drop, width = 0.8)
	plt.xlabel('Consecutive NaN')
	plt.ylabel('Number of trials')

	fig.add_subplot(2,2,2)
	non_zero_nans_trial = list(nan_per_trial)
	non_zero_nans_trial.remove(0)
	plt.hist(non_zero_nans_trial, bins=40)
	plt.xlabel('Number of NaN')

	plt.savefig(os.path.join(paths.path2Figures, 'nan_ivestigation.png'))



def normalized_intensity(acti, interpolated_acti):
	N, T, K = interpolated_acti.shape
	# Illustration with some well chosen example
	nan_per_trial = pd.Series(np.isnan(acti[0,:,:]).sum(axis = 0), index = range(K))
	roi = nan_per_trial.sort_values(ascending = False).index[1]

	fig = plt.figure(figsize=(10,8))

	plt.plot(interpolated_acti[0,:,roi],lw = 1, color = 'red', label = 'Interpolated')
	plt.plot(acti[0,:,roi], lw = 1, color = 'blue', label = 'Raw')
	plt.xlabel('Timeframe')
	plt.ylabel('Normalized fluorescence intensity')
	plt.legend()

	plt.savefig(os.path.join(paths.path2Figures, 'normalized_intensity.png'))


def explore_integrity(interpolated_acti):
	N, T, K = interpolated_acti.shape
	nb_flag_roi = []
	nb_flag_trial = []

	for tol in range(50):
	    max_dist = np.nanmax(interpolated_acti, axis = 1)
	    max_mask = max_dist > tol

	    flag_roi = []
	    flag_trial = []
	    big_flag = max_mask * max_dist
	    
	    for roi in range(N):
	        for trial in range(K):
	            if max_mask[roi, trial]:
	                flag_roi.append(roi)
	                flag_trial.append(trial)
	                
	    flag_roi = list(set(flag_roi))
	    flag_trial = list(set(flag_trial))
	    nb_flag_roi.append(len(flag_roi))
	    nb_flag_trial.append(len(flag_trial))

	fig = plt.figure(figsize=(10,5))

	fig.add_subplot(1,2,1)
	plt.plot(np.arange(50), nb_flag_roi)
	plt.title('ROI lost')
	plt.xlabel('Threshold')
	plt.ylabel('Number of ROI')

	fig.add_subplot(1,2,2)
	plt.plot(np.arange(50), nb_flag_trial)
	plt.title('Trials lost')
	plt.xlabel('Threshold')
	plt.ylabel('Number of trials')
	plt.savefig(os.path.join(paths.path2Figures, 'explore_integrity1.png'))


	fig = plt.figure(figsize = (25,15))

	fig.add_subplot(2,1,1)
	sns.boxplot(x = np.arange(K), y = [max_dist[:,i] for i in range(K)])
	plt.xticks(np.arange(K)[::40],  np.arange(K)[::40], rotation = 'horizontal')
	plt.xlabel('Trials', {'fontsize': 'large', 'fontweight' : 'roman'})
	plt.ylabel('Distribution of maxima across ROI', {'fontsize': 'large', 'fontweight' : 'roman'})

	fig.add_subplot(2,1,2)
	plt.imshow(big_flag, cmap = 'hot')
	plt.xticks(np.arange(K)[::40],  np.arange(K)[::40], rotation = 'horizontal')
	plt.xlabel('Trials', {'fontsize': 'large', 'fontweight' : 'roman'})
	plt.ylabel('ROI', {'fontsize': 'large', 'fontweight' : 'roman'})

	plt.savefig(os.path.join(paths.path2Figures, 'explore_integrity2.png'))

def potential_outliers(f0, flags, acti):
	N, T, K = acti.shape
	fig = plt.figure(figsize = (20,10))

	fig.add_subplot(1,2,1)
	plt.imshow(f0, cmap='hot')
	plt.xticks(np.arange(K)[::40],  np.arange(K)[::40], rotation = 'horizontal')
	plt.xlabel('Trials', {'fontsize': 'large', 'fontweight' : 'roman'})
	plt.ylabel('Distribution of maxima across ROI', {'fontsize': 'large', 'fontweight' : 'roman'})

	fig.add_subplot(1,2,2)
	plt.imshow(flags, cmap = 'hot')
	plt.xticks(np.arange(K)[::40],  np.arange(K)[::40], rotation = 'horizontal')
	plt.xlabel('Trials', {'fontsize': 'large', 'fontweight' : 'roman'})
	plt.ylabel('ROI', {'fontsize': 'large', 'fontweight' : 'roman'})

	plt.savefig(os.path.join(paths.path2Figures, 'potential_outliers.png'))

def explore_unreal_roi(big_flag, roi_tensor):
	blob = big_flag.sum(axis=1)
	plt.imshow(tca.make_map(roi_tensor, blob), cmap='hot')
	plt.savefig(os.path.join(paths.path2Figures, 'explore_unreal_roi.png'))

def plot_flagged_roi(f0, flag_roi):
	width = 30
	height = 20
	assert width * height >= len(flag_roi), 'Increase number of subplots'
	fig = plt.figure(figsize = (40,20))
	for i, roi in enumerate(flag_roi):
	    fig.add_subplot(width, height, i+1)
	    plt.plot(f0[roi,:])
	plt.savefig(os.path.join(paths.path2Figures, 'plot_flagged_roi.png'))


def behaviogram(raw_beh, trials_to_drop):
	'''function in contruction'''
	raw_beh_mat = raw_beh['ALLBEHmatrix']
	learn_color, limit_block, limit_day = tca.behaviorgram(raw_beh_mat, trials_to_drop)