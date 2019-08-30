import os
import torch
import random
import numpy as np

from functions import data
from functions import settings as sett 
from functions import TCA as t

# Set backend to pytorch as default is numpy 
# tl.set_backend('pytorch')

# Set default tensor type to cuda tensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load parameters 
params = sett.params()
paths = sett.paths()
ar = sett.arguments()

# Load arguments
args = ar.get_arguments()
selection = ar.get_selection()
tca_sett = ar.get_tca_sett()
animal = ar.get_animal()

# Load processed data
meta_df, roi_tensor, acti, norm_acti, smoothed_acti = data.load_processed_data_all(animal)
meta_df, acti, norm_acti, smoothed_acti = data.select_data(meta_df, acti, norm_acti, smoothed_acti, selection)

# Define a name for the simulation based on selection criteria
name = ''.join(['_' + k + '-' + str(selection[k]) for k in selection if not selection[k] == None])[1:]

# Define specific paths for saving
path = os.path.join(paths.path2Output, animal, args.function, args.init, str(args.rank), name)
path_fig = os.path.join(paths.path2Figures, animal, args.function, args.init, str(args.rank), name)

# Create directories if they don't exist
for p in [path, path_fig]:	
	try:
		os.makedirs(p)
	except:
		FileExistsError

# Define a TCA object
model = t.TCA(function=args.function, rank=args.rank, init=args.init, verbose=args.verbose, roi_tensor=roi_tensor, ff=args.fixed_factor)

# Fit the TCA model on activity data 
factors = model.fit(torch.tensor(norm_acti))

# Print three factorplots, each one with a different trial colouring
model.factorplot(meta_df, animal, name + 'odor', path_fig, color=meta_df['Odor Color'].tolist(), balance=True, order=False)
model.factorplot(meta_df, animal, name + 'rew', path_fig, color=meta_df['Reward Color'].tolist(), balance=True, order=False)
model.factorplot(meta_df, animal, name + 'behavior', path_fig, color=meta_df['Behavior Color'].tolist(), balance=True, order=False)

# Get fit error for saving. model.detailed_error() would give supplementary information
rec_errors = model.error()

# Perform random forest prediction with behavioral data
score_odor, score_rew, clf_odor, clf_rew = model.predict(meta_df, 50)

# Generate figures by feature importance and save them
# model.important_features_map(path_fig)
# model.important_features_time(path_fig)

# Choose best ROIs based on each ROI variance
# feat_odor, feat_rew, reshaped_odor, reshaped_rew = model.best_predictive_rois(path_fig)

# Redefine tensor mask to match the new number of ROIs
# model.roi_tensor = roi_tensor[:, :, feat_odor]

# Fit the same model again on this restricted, best ROIs, data
# model.fit(torch.tensor(norm_acti[feat_odor, :, :]))

# Print factorplot
# model.factorplot(meta_df, animal, name + 'reduce_roi', path_fig, color=meta_df['Odor Color'].tolist(), balance=True, order=False)

# Perform random forest prediction on bets predictive ROIs' data
# score_odor, score_rew, clf_odor, clf_rew = model.predict(meta_df)

# Compute a reward coefficient accounting for the disparity between classes and modify score
reward_balance = max(len(meta_df[meta_df.Reward == 0]), len(meta_df[meta_df.Reward == 1])) / len(meta_df['Reward'])
score_rew = 0.5 + ((score_rew - reward_balance) * 0.5) / (1 - reward_balance)

# Print results if verbose mode enabled
if args.verbose: 
	print("Odor prediction - Accuracy: %0.4f" % (score_odor))
	print("Reward prediction - Accuracy: %0.4f" % (score_rew))

# Save all results in Output folder
data.save_results(factors, rec_errors, score_odor, score_rew, name, path)


