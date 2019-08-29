import os
import torch
import random
import numpy as np
import pandas as pd

from functions import data
from functions import settings as sett 
from functions import TCA as t

# tl.set_backend('pytorch')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load parameters 
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

# if args.smooth:
# 	norm_acti = smoothed_acti

name = ''.join(['_' + k + '-' + str(selection[k]) for k in selection if not selection[k] == None])[1:]

path = os.path.join(paths.path2Output, animal, args.function, args.init, str(args.rank), name)
path_fig = os.path.join(paths.path2Figures, animal, args.function, args.init, str(args.rank), name)

for p in [path, path_fig]:	
	try:
		os.makedirs(p)
	except:
		FileExistsError

model = t.TCA(function=args.function, rank=args.rank, init=args.init, verbose=args.verbose, roi_tensor=roi_tensor, ff=args.fixed_factor)
factors = model.fit(torch.tensor(norm_acti))
model.factorplot(meta_df, animal, name, path_fig, color=meta_df['Odor Color'].tolist(), balance=True, order=False)
rec_errors = model.error()


score_odor, score_rew, clf_odor, clf_rew = model.predict(meta_df, 50)

model.important_features_map(path_fig)
model.important_features_time(path_fig)

feat_odor, feat_rew, reshaped_odor, reshaped_rew = model.best_predictive_rois(path_fig)

model.roi_tensor = roi_tensor[:, :, feat_odor]
model.fit(torch.tensor(norm_acti[feat_odor, :, :]))
model.factorplot(meta_df, animal, name + 'reduce_roi', path_fig, color=meta_df['Odor Color'].tolist(), balance=True, order=False)

# score_odor, score_rew, clf_odor, clf_rew = model.predict(meta_df)

data.save_results(factors, rec_errors, score_odor, score_rew, name, path)

if args.verbose: 
	print("Odor prediction - Accuracy: %0.4f" % (score_odor))
	print("Reward prediction - Accuracy: %0.4f" % (score_rew))

