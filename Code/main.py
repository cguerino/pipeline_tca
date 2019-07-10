import os
import argparse
import numpy as np
import pandas as pd

from functions import plot
from functions import data
from functions import preprocessing as prepro
from functions import settings as sett 

params = sett.params()

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

args = parser.parse_args()

# Choose an animal to process
animal = params.animal_list[args.animal]

# Loading data from folder
meta_df, roi_tensor, acti, beh_mat, f0 = data.load_data(animal, args.verbose)

if args.plotting: plot.map_of_rois(acti, roi_tensor)

# Define trials to drop for too much NaNs or consecutive NaNs
trials_to_drop = prepro.trials_to_drop(acti, args.cutoff, args.thres, args.verbose)

# Drop such trials
acti = np.delete(acti, trials_to_drop, 2)
f0_fin = np.delete(f0, trials_to_drop, 1)
beh_mat = np.delete(beh_mat, trials_to_drop, 0)

if args.verbose: print('We delete {0} trials because of NaN'.format(len(trials_to_drop)))

# Replace missing data by interpolation
interpolated_acti, drop_roi = prepro.interpolation(acti, args.verbose)

# Plots for data exploration
if args.plotting: plot.potential_outliers(f0, flags, acti)
if args.plotting: plot.normalized_intensity(acti, interpolated_acti)
if args.plotting: plot.explore_integrity(interpolated_acti)
if args.plotting: plot.explore_unreal_roi(bigflag, roi_tensor)
if args.plotting: plot.plot_flagged_roi(flag_roi)

# Delete remaining ROIs that were not affected by interpolation 
interpolated_acti = np.delete(interpolated_acti, drop_roi, 0)
roi_tensor = np.delete(roi_tensor, drop_roi, 2)

if args.verbose: print('We delete {0} ROI because of NaN'.format(len(drop_roi)))

# Flag roi that still contain NaNs
flag_roi, flag_trial, bigflag = prepro.clean_messy_ROIs(interpolated_acti, args.tol)

# Delete such ROIs
interpolated_acti = np.delete(interpolated_acti, flag_roi, 0)
roi_tensor = np.delete(roi_tensor, flag_roi, 2)

if args.verbose: print('We delete {0} flagged ROI'.format(len(flag_roi)))

# Generate color matrix for later plotting
color_df = data.generate_color_df(beh_mat)

# if args.plotting: plot.behaviorgram(raw_beh, trials_to_drop)

# Save processed data in a folder
data.save_data(interpolated_acti, flag_roi, trials_to_drop, roi_tensor, color_df, animal)

