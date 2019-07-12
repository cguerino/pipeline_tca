import os
import argparse
import numpy as np
import pandas as pd

import tensorly as tl
from functions import plot
from functions import data
from functions import preprocessing as prepro
from functions import settings as sett 
from functions import tca_utils as tca

from tensorly.decomposition import non_negative_parafac, parafac, custom_parafac

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import torch

import itertools

tl.set_backend('pytorch')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

params = sett.params()
paths = sett.paths()

# Create parser and list all arguments
parser = argparse.ArgumentParser(description='Parameters for computing')

parser.add_argument('--plotting', '-p', action='store_true', 
					help='Plot figures during processing')
parser.add_argument('--verbose', '-v', action='store_true', 
					help='Display detailed output in prompt')
parser.add_argument('--cutoff', '-co', type=int, default=25, 
					help='Cut-off for consecutive NaNs in a trial')
parser.add_argument('--thres', '-th', type=int, default=80, 
					help='Threshold for non_consecutive NaNs in a trial')
parser.add_argument('--animal', '-a', type=int, default=0, 
					help='Choose animal to process')
parser.add_argument('--tol', '-tl', type=int, default=15, 
					help='Tolerance for data')
parser.add_argument('--preprocess', '-pre', action='store_true', 
					help='Perform preprocessing')
parser.add_argument('--computation', '-com', action='store_true', 
					help='Perform computation')
parser.add_argument('--rank', '-r', type=int, default=6, 
					help='Rank of the TCA')
parser.add_argument('--function', '-f', type=str, default='custom_parafac', 
					help='TCA function to use: parafac, non_negative_parafac, custom_parafac')

# Select relevant data for processing
parser.add_argument('--Block', '-B', type=int, nargs='*', default=None, 
					help='Blocks to choose from. All blocks are taken by default')
parser.add_argument('--Odor', '-O', type=str, nargs='*', default=None, 
					help='Odors to choose from. All odors are taken by default')
parser.add_argument('--Stimulus', '-S', type=str, nargs='*', default=None, 
					help='Stimulus to choose from. All stimulus are taken by default')
parser.add_argument('--Behavior', '-BE', type=str, nargs='*', default=None, 
					help='Behaviors to choose from. All behaviors are taken by default')
parser.add_argument('--Reward', '-R', type=int, default=None, 
					help='Rewards to choose from. All rewards are taken by default')
parser.add_argument('--Date', '-D', type=str, nargs='*', default=None, 
					help='Dates to choose from. All dates are taken by default')
parser.add_argument('--Day', '-DY', type=int, nargs='*', default=None, 
					help='Days to choose from. All days are taken by default')
parser.add_argument('--Learnstate', '-L', type=int, default=None, 
					help='Learning states to choose from. All learning states are taken by default')
parser.add_argument('--Expclass', '-E', type=int, nargs='*', default=None, 
					help='Experiment classes to choose from. All experiment classes are taken by default')
parser.add_argument('--Performance', '-P', type=int, nargs='*', default=None, 
					help='Performances to choose from. All performances are taken by default')

args = parser.parse_args()

animal = params.animal_list[args.animal]

#################################### PREPROCESSING ##########################################
if args.preprocess:
	selection = {
		'Block':args.Block, 
		'Odor': args.Odor,
		'Stimulus': args.Stimulus,
		'Behavior': args.Behavior,
		'Reward': args.Reward,
		'Date': args.Date,
		'Day': args.Day,
		'Learning State': args.Learnstate,
		'Experiment Class': args.Expclass,
		'Performance': args.Performance
	}

	# Loading data from folder
	meta_df, roi_tensor, acti, beh_mat, f0, trials_of_interest = data.load_data(animal, 
																			selection, args.verbose)

	if args.plotting: plot.map_of_rois(acti, roi_tensor)

	# Define trials to drop for too much NaNs or consecutive NaNs
	trials_to_drop = prepro.trials_to_drop(acti, args.cutoff, args.thres, args.verbose)

	# Drop such trials
	acti = np.delete(acti, trials_to_drop, 2)
	f0_fin = np.delete(f0, trials_to_drop, 1)
	beh_mat = np.delete(beh_mat, trials_to_drop, 0)
	meta_df = meta_df.drop([meta_df.index[i] for i in trials_to_drop])

	if args.verbose: print('We delete {0} trials because of NaN'.format(len(trials_to_drop)))

	# Replace missing data by interpolation
	interpolated_acti, drop_roi = prepro.interpolation(acti, args.verbose)

	# Plots for data exploration
	if args.plotting: plot.normalized_intensity(acti, interpolated_acti)
	if args.plotting: plot.explore_integrity(interpolated_acti)

	# Delete remaining ROIs that were not affected by interpolation 
	interpolated_acti = np.delete(interpolated_acti, drop_roi, 0)
	roi_tensor = np.delete(roi_tensor, drop_roi, 2)

	if args.verbose: print('We delete {0} ROI because of NaN'.format(len(drop_roi)))

	# Flag roi that still contain NaNs
	flag_roi, flag_trial, flags = prepro.clean_messy_ROIs(interpolated_acti, args.tol)

	# Plots to explore
	if args.plotting: plot.potential_outliers(f0, flags, acti)
	if args.plotting: plot.explore_unreal_roi(flags, roi_tensor)
	if args.plotting: plot.plot_flagged_roi(f0, flag_roi)

	# Delete such ROIs
	interpolated_acti = np.delete(interpolated_acti, flag_roi, 0)
	roi_tensor = np.delete(roi_tensor, flag_roi, 2)

	if args.verbose: print('We delete {0} flagged ROI'.format(len(flag_roi)))

	# Save processed data in a folder
	data.save_data(interpolated_acti, flag_roi, trials_to_drop, roi_tensor, meta_df, animal)

#################################### COMPUTATION ##########################################
if args.computation:
	meta_df, roi_tensor, acti = data.load_processed_data(animal)

	# To move in preprocessing steps
	norm_acti = prepro.normalize_acti(acti)

	assert roi_tensor.shape[2] == norm_acti.shape[0]
 	
	norm_acti = torch.tensor(norm_acti)

	factors, rec_errors = tca.custom_parafac(norm_acti, args.rank, n_iter_max=10000, tol=1e-07, 
											  verbose=1, return_errors=True, neg_fac=1)

	factors = [f.cpu().numpy() for f in factors]

	X, y_odor, y_rew = shuffle(factors[2], meta_df['Odor'].tolist(), meta_df['Reward'].tolist())
	clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, max_features='sqrt')
	X = StandardScaler().fit_transform(X)
	scores_odor = cross_val_score(clf, X, y_odor, cv = 8)
	scores_rew = cross_val_score(clf, X, y_rew, cv = 8)

	print("Odor prediction - Accuracy: %0.2f (+/- %0.2f)" % (scores_odor.mean(), scores_odor.std()))
	print("Reward prediction - Accuracy: %0.2f (+/- %0.2f)" % (scores_rew.mean(), scores_rew.std()))

	tca.factorplot(factors, roi_tensor, meta_df, color=meta_df['Odor Color'].tolist(), balance=True)



