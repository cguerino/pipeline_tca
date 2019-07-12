import itertools
import numpy as np 
import pandas as pd


def trials_to_drop(acti, cutoff, thres, verbose):
	N, T, K = acti.shape
	df = pd.DataFrame(acti[0,:,:])
	df = df.fillna(-2)
	cons_nan_per_trial = [max(sum(1 for x in j if x == -2) for i, j in itertools.groupby(df[trial])) for trial in df]
	drop_trial = [trial for trial in range(K) if cons_nan_per_trial[trial] > cutoff]
	if verbose: print('With a cut-off of {0} consecutive NaN, we will drop {1} trials'.format(cutoff, len(drop_trial)))

	nan_per_trial = pd.Series(np.isnan(acti[0,:,:]).sum(axis=0))
	drop_tot = nan_per_trial[nan_per_trial > thres].index.tolist()
	if verbose: print('We will drop {0} for having more than {1} NaN'.format(len(drop_tot), thres))

	drop_trial = list(set(drop_trial + drop_tot))
	drop_trial.sort()
	if verbose: print('This represents', len(drop_trial), 'trials out of', K)

	return drop_trial

def interpolation(acti, verbose):
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


def behaviogram(raw_beh, trials_to_drop):
	'''function in contruction'''
	b = tca.behaviorgram(raw_beh_mat, trials_to_drop)

	return b

def normalize_acti(acti):
	N, T, K = acti.shape
	norm_acti = np.zeros_like(acti)
	# Get the maximum amplitude across neurons and across time
	max_amp = np.nanmax(acti, axis = (0,1))
	for t in range(T):
	    for n in range(N):
	        norm_acti[n,t,:] = np.divide(acti[n,t,:] + 1, max_amp + 1)

	return norm_acti