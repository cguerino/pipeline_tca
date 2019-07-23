import itertools
import numpy as np 
import pandas as pd


def trials_to_drop(acti, cutoff, thres, verbose=False):
	""" Compute a list of trial indexes that do not match exclusion criteria 
	Arguments:
		acti {array} -- activation tensor
		cutoff {int} -- consecutive NaNs limit
		thres {int} -- threshold for number of NaNs in a single trial


	Keyword Arguments:
		verbose {bool} -- enable verbose mode
	Returns:
		trials_to_drop {list} -- non-conform trial indexes
	"""
	N, T, K = acti.shape
	df = pd.DataFrame(acti[0,:,:])
	df = df.fillna(-2)
	cons_nan_per_trial = [max(sum(1 for x in j if x == -2) for i, j in itertools.groupby(df[trial])) for trial in df]
	drop_cons = [trial for trial in range(K) if cons_nan_per_trial[trial] > cutoff]
	if verbose: print('With a cut-off of {0} consecutive NaN, we will drop {1} trials'.format(cutoff, len(drop_cons)))

	nan_per_trial = pd.Series(np.isnan(acti[0,:,:]).sum(axis=0))
	drop_tot = nan_per_trial[nan_per_trial > thres].index.tolist()
	if verbose: print('We will drop {0} trials for having more than {1} NaN'.format(len(drop_tot), thres))

	trials_to_drop = list(set(drop_cons + drop_tot))
	trials_to_drop.sort()
	if verbose: print('This represents', len(trials_to_drop), 'trials out of', K)

	return trials_to_drop

def interpolation(acti, verbose):
	""" Interpolate data to fill blank left by consecutive NaNs under cutoff

	Arguments:
		acti {array} -- activation tensor

	Keywords Arguments:
		verbose {bool} -- enable verbose mode

	Returns:
		line_acti {array} -- activation matrix with linear interpolation
		drop_roi {list} -- list of ROIs containing NaNs
	"""
	N, T, K = acti.shape
	line_acti = np.zeros_like(acti)
	
	drop_roi = []
	for roi in range(N):
		df = pd.DataFrame(acti[roi,:,:])
		df_line_interp = df.interpolate(method = 'linear', axis=0, limit=25, limit_direction='both')
		
		if df_line_interp.isna().sum().sum() > 0:
			print("For ROI {0}, there are {1} NaN".format(roi, df_line_interp.isna().sum().sum()))
			drop_roi.append(roi)
		
		line_acti[roi,:,:] = df_line_interp.values
	
	return line_acti, drop_roi

def clean_messy_ROIs(interpolated_acti, tol):
	""" Detect incoherence in activation values for each ROI

	Arguments:
		interpolated_acti {array} -- activation tebsor with linear interpolation correction
		tol {int} -- tolerance value

	Keywords Arguments:
		None

	Returns:
		flag_roi {list} -- ROIs to remove
		flag_trail {list} -- trials to remove
		big_flag {array} -- mask of flagged values
	"""
	N, T, K = interpolated_acti.shape
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
	
	return flag_roi, flag_trial, big_flag


def norm(acti):
	""" Normalize activation data

	Arguments:
		acti {array} -- activation tensor

	Keywords Arguments:
		None

	Returns:
		None
	"""
	N, T, K = acti.shape
	norm_acti = np.zeros_like(acti)
	# Get the maximum amplitude across neurons and across time
	max_amp = np.nanmax(acti, axis = (0,1))
	for t in range(T):
		for n in range(N):
			norm_acti[n,t,:] = np.divide(acti[n,t,:] + 1, max_amp + 1)

	#norm_acti = (acti - np.min(acti)) / (np.max(acti) - np.min(acti))*2 - 1
	return norm_acti

def min_max(acti):
	norm_acti = (acti - np.min(acti)) / (np.max(acti) - np.min(acti))

	return norm_acti

def day(acti, meta_df):
	for day in set(meta_df['Day'].tolist()):
		trials_to_norm = meta_df[meta_df['Day'] ==  day].index.tolist()
		acti[:, : , trials_to_norm] = norm(acti[:, : , trials_to_norm])

	return acti

