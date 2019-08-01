import os
import time
import torch
import random
import numpy as np
import pandas as pd
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

tl.set_backend('pytorch')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

params = sett.params()
paths = sett.paths()
args = sett.arguments()
animal = sett.get_animal()



preprocess_sett = {
	'Cut_off': args.cutoff,
	'Threshold': args.thres,
	'Animal': args.animal,
	'Tol': args.tol,
	'Norm': args.norm
}

selection = {
}

# Loading data from folder
meta_df, roi_tensor, acti, f0, trials_of_interest = data.load_data(animal, 
																   selection,
																   args.verbose)

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

# Normalize data
if args.norm == 'norm':
	norm_acti = prepro.norm(interpolated_acti)
elif args.norm == 'min_max':
	norm_acti = prepro.min_max(interpolated_acti)
elif args.norm == 'day':
	norm_acti = prepro.day(interpolated_acti, meta_df)

smoothed_acti = np.empty_like(interpolated_acti)

for N in tqdm(range(interpolated_acti.shape[0]), ascii=True):
	for K in range(interpolated_acti.shape[2]):
		smoothed_acti[N, :, K] = tca.curve_smoothing(norm_acti[N, :, K])
			
# Save processed data in a folder
data.save_data_all(interpolated_acti, norm_acti, smoothed_acti, flag_roi, roi_tensor, meta_df, animal, preprocess_sett)
