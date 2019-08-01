import os
import torch
import random
import numpy as np
import pandas as pd
import tensorly as tl

from functions import data
from functions import settings as sett 
from functions import tca_utils as tca
from functions import training as train

tl.set_backend('pytorch')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

params = sett.params()
paths = sett.paths()
ar = sett.arguments()

args = ar.get_arguments()
selection = ar.get_selection()
tca_sett = ar.get_tca_sett()
animal = ar.get_animal()

# Load processed data
meta_df, roi_tensor, acti, norm_acti, smoothed_acti = data.load_processed_data_all(animal)
meta_df, acti, norm_acti, smoothed_acti = data.select_data(meta_df, acti, norm_acti, smoothed_acti, selection)

name = '{}'.format(random.getrandbits(32))

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
	factors, rec_errors = train.oob_error_rank(norm_acti, meta_df, args.init, animal, tca_sett, name)

# Bring back factors array to cpu memory and convert it to numpy array
factors_tensor = tl.kruskal_tensor.kruskal_to_tensor(factors).cpu().numpy()
factors = [f.cpu().numpy() for f in factors]


# Perform random forest classification
score_odor, score_rew, clf_odor, clf_rew = train.rf_oob_score(factors, meta_df, 50)
train.feature_importance_roi_maps(factors, roi_tensor, clf_odor, animal, tca_sett, name, spe='odor')
train.feature_importance_time_factor(factors, roi_tensor, clf_odor, animal, tca_sett, name, spe='odor')

train.feature_importance_roi_maps(factors, roi_tensor, clf_rew, animal, tca_sett, name, spe='rew')
train.feature_importance_time_factor(factors, roi_tensor, clf_rew, animal, tca_sett, name, spe='rew')

best_rois_idx, reshaped_best_rois = train.best_odor_predictive_rois(factors, roi_tensor, clf_odor, animal, tca_sett, name)
best_rew_idx, reshaped_best_rew = train.best_rew_predictive_rois(factors, roi_tensor, clf_rew, animal, tca_sett, name)

data.save_results(factors, rec_errors, score_odor, score_rew, animal, name, tca_sett)

if args.verbose: 
	print("Odor prediction - Accuracy: %0.4f" % (score_odor))
	print("Reward prediction - Accuracy: %0.4f" % (score_rew))

# Plot data and save it
tca.factorplot(factors, roi_tensor, meta_df, animal, name, selection, tca_sett, color=meta_df['Reward Color'].tolist(), balance=True)

# tca.get_raw_traces(best_rois_idx, acti)
######################### SECOND PASS ##############################
norm_acti = torch.tensor(norm_acti[best_rois_idx, :, :])

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
	factors, rec_errors = train.oob_error_rank(norm_acti, meta_df, args.init, animal, tca_sett, name)

# Bring back factors array to cpu memory and convert it to numpy array
factors_tensor = tl.kruskal_tensor.kruskal_to_tensor(factors).cpu().numpy()
factors = [f.cpu().numpy() for f in factors]


# Perform random forest classification
score_odor, score_rew, clf_odor, clf_rew = train.rf_oob_score(factors, meta_df, 50)
roi_tensor = roi_tensor[:, :, best_rew_idx]

if args.verbose: 
	print("Odor prediction - Accuracy: %0.4f" % (score_odor))
	print("Reward prediction - Accuracy: %0.4f" % (score_rew))

# Plot data and save it
tca.factorplot(factors, roi_tensor, meta_df, animal, name + '2', selection, tca_sett, color=meta_df['Behavior Color'].tolist(), balance=True)

# scores = tca.compute_r2_score(norm_acti, factors_tensor)


