import os
import torch
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf 
import tensorly as tl
from functions import plot
from functions import data
from functions import training as train
from functions import preprocessing as prepro
from functions import settings as sett 
from functions import tca_utils as tca
import matplotlib.pyplot as plt 

from tqdm import tqdm
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


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
parser.add_argument('--animal', '-a', type=int, default=15, 
					help='Choose animal to process')
parser.add_argument('--tol', '-tl', type=int, default=15, 
					help='Tolerance for data')
parser.add_argument('--preprocess', '-pre', action='store_true', 
					help='Perform preprocessing')
parser.add_argument('--computation', '-com', action='store_true', 
					help='Perform computation')
parser.add_argument('--rank', '-r', type=int, default=6, 
					help='Rank of the TCA')
parser.add_argument('--function', '-f', type=str, default='non_negative_parafac', 
					help='TCA function to use: parafac, non_negative_parafac, custom_parafac')
parser.add_argument('--neg_fac', '-nf', type=str, default=0, 
					help='Factor of the TCA which is allowed to be negative')
parser.add_argument('--tmp', '-tmp', action='store_true', 
					help='Save and compute data of tmp folder for debugging')
parser.add_argument('--init', '-i', type=str, default='random', 
					help='Initialization method for parafac')
parser.add_argument('--norm', '-n', type=str, default='norm', 
					help='Normalization_method')

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

if not args.tmp: 
	name = '{}'.format(random.getrandbits(32))
else:
	name = 'tmp'

if not args.preprocess and not args.computation:
	args.preprocess, args.computation = True, True

arguments = {
	'Cut_off': args.cutoff,
	'Threshold': args.thres,
	'Animal': args.animal,
	'Tol': args.tol,
	'Rank': args.rank,
	'Function': args.function,
	'Neg_Fac': args.neg_fac, 
	'Init':args.init,
	'Norm': args.norm
}

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

#################################### PREPROCESSING ##########################################
if args.preprocess:

	# Loading data from folder
	meta_df, roi_tensor, acti, f0, trials_of_interest = data.load_data(animal, 
																			selection, args.verbose)

	if args.plotting: plot.map_of_rois(acti, roi_tensor)

	# Define trials to drop for too much NaNs or consecutive NaNs
	trials_to_drop = prepro.trials_to_drop(acti, args.cutoff, args.thres, args.verbose)

	# Drop such trials
	acti = np.delete(acti, trials_to_drop, 2)
	f0_fin = np.delete(f0, trials_to_drop, 1)
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
	data.save_data(interpolated_acti, flag_roi, trials_to_drop, roi_tensor, meta_df, animal, name, arguments, selection)

#################################### COMPUTATION ##########################################
if args.computation:
	if args.tmp:
		name = 'tmp'
	# Load processed data
	meta_df, roi_tensor, acti = data.load_processed_data(animal, name, arguments)

	if args.tmp:
		name = '{}'.format(random.getrandbits(32))

	# Normalize data
	if args.norm == 'norm':
		norm_acti = prepro.norm(acti)
	elif args.norm == 'min_max':
		norm_acti = prepro.min_max(acti)
	elif args.norm == 'day':
		norm_acti = prepro.day(acti, meta_df)
	else:
		norm_acti = acti

	# Convert to pytorch tensor for computation 	
	norm_acti = torch.tensor(norm_acti)

	# Choose with which function compute TCA
	if args.function == 'custom_parafac':
		if args.neg_fac != None:
			factors, rec_errors = tca.custom_parafac(norm_acti, args.rank, n_iter_max=10000, tol=1e-07, 
											  verbose=1, return_errors=True, neg_fac=args.neg_fac, init=args.init)
	elif args.function == 'parafac':
		factors, rec_errors = tl.decomposition.parafac(norm_acti, args.rank, n_iter_max=10000, tol=1e-10, 
											  verbose=1, return_errors=True, init=args.init)
	elif args.function == 'non_negative_parafac':
		factors, rec_errors = tl.decomposition.non_negative_parafac(norm_acti, args.rank, n_iter_max=10000, tol=1e-07, 
											  verbose=1, return_errors=True, init=args.init)
	elif args.function == 'test_rank':
		factors, rec_errors = train.oob_error_rank(norm_acti, meta_df, args.init, animal, arguments, name)
	elif args.function == 'speed':
		rec_errors = []
		factors = train.speed_tca(norm_acti)

	# Bring back factors array to cpu memory and convert it to numpy array
	factors_tensor = tl.kruskal_tensor.kruskal_to_tensor(factors).cpu().numpy()
	factors = [f.cpu().numpy() for f in factors]



	# Perform random forest classification
	score_odor, score_rew, clf_odor, clf_rew = train.rf_oob_score(factors, meta_df, 50)
	train.feature_importance_roi_maps(factors, roi_tensor, clf_odor, animal, arguments, name + 'odor')
	train.feature_importance_time_factor(factors, roi_tensor, clf_odor, animal, arguments, name + 'odor')
	
	train.feature_importance_roi_maps(factors, roi_tensor, clf_rew, animal, arguments, name + 'rew')
	train.feature_importance_time_factor(factors, roi_tensor, clf_rew, animal, arguments, name + 'rew')
	

	# Save data 
	data.save_results(factors, rec_errors, score_odor, score_rew, animal, name, arguments)

	# if args.verbose: 
	# 	print("Odor prediction - Accuracy: %0.4f (+/- %0.4f)" % (scores_odor.mean(), score_odor.std()))
	# 	print("Reward prediction - Accuracy: %0.4f (+/- %0.4f)" % (scores_rew.mean(), score_rew.std()))

	# Plot data and save it
	tca.factorplot(factors, roi_tensor, meta_df, animal, name, selection, arguments, color=meta_df['Odor Color'].tolist(), balance=True)

	# scores = tca.compute_r2_score(norm_acti, factors_tensor)


