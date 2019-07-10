import os
import numpy as np
import pandas as pd
from hdf5storage import loadmat

from functions import settings as sett 
params = sett.params()
paths = sett.paths()


def load_data(animal, selection, verbose):
	path_raw = os.path.join(paths.path2Data, animal, 'Raw Data')
	
	raw_paths = [os.path.join(path_raw, file) for file in ['ROIBinmaps_{0}.mat'.format(animal), 'ALLDFF.mat', 'ALLBEHmatrix.mat', 'ALLf0.mat']]

	meta_df = pd.read_csv(os.path.join(path_raw, 'meta_df.csv'), index_col=0)

	for param in selection:
		print(param)
		if selection[param] != None:
			meta_df = meta_df[meta_df[param].isin(selection[param])]

	trials_of_interest = meta_df.index.tolist()

	if not trials_of_interest:
		raise ValueError('Selection is too restrictive, please select at least 1 trial')

	# Load matrices as dictionnary {name: matrix_values}
	roi_file, raw_mat, raw_beh, f0 = list(map(loadmat, raw_paths))
	
	# Indexing data of interest
	roi_tensor = roi_file['ROIBinmaps']

	# Remove the first second of acquisition
	acti = raw_mat['ALLDFF'][:,15:,:]
	acti = acti[:,:,trials_of_interest]

	beh_mat = raw_beh['ALLBEHmatrix']
	beh_mat = beh_mat[trials_of_interest]
	
	f0 = f0['F0ALL']
	f0 = f0[:,trials_of_interest]

	# Check # of trial consistency
	assert acti.shape[2] == beh_mat.shape[0]

	# Check # of ROI consistency
	assert acti.shape[0] == roi_tensor.shape[2]

	N, T, K = acti.shape
	if verbose: print('Data contain activity of', N, 'ROI, on a time scale of', T/15, 'seconds, during', K, 'trials')

	return meta_df, roi_tensor, acti, beh_mat, f0, trials_of_interest

def save_data(interpolated_acti, flag_roi, drop_trial, roi_tensor, color_df, animal):
	# assert beh_mat.shape[0] + len(drop_trial) == raw_beh_mat.shape[0]
	# assert b[1][-1][1] == beh_mat.shape[0]

	path = os.path.join(paths.path2Output, animal)
	try:
	    os.mkdir(path)
	except:
	    FileExistsError

	np.save(os.path.join(path, 'acti'), interpolated_acti)
	np.save(os.path.join(path,'flag_roi'), flag_roi)
	np.save(os.path.join(path,'drop_trial'), drop_trial)
	np.save(os.path.join(path,'roi_tensor'), roi_tensor)
	# np.save(os.path.join(path,'behaviorgram'), b)
	color_df.to_csv(os.path.join('color_df.csv'))

def generate_color_df(beh_mat):
	beh_mat = beh_mat.astype(int)

	odor = beh_mat[:,0]
	beh = beh_mat[:,1]
	learning = beh_mat[:,2]
	day = beh_mat[:,3]

	color_list = params.color_list

	color_lea = [color_list[i] for i in learning]
	color_day = [color_list[i] for i in day]
	color_odor = [color_list[i+1] for i in odor]
	color_beh = [color_list[i+1] for i in beh]
	color_rew = [color_list[i+7] for i in beh]

	color_dict = {'Behavior' : color_beh, 'Day' : color_day, 'Learning' : color_lea, 
				  'Odor' : color_odor, 'Reward' : color_rew}
	
	color_df = pd.DataFrame(data = color_dict)

	return color_df
	raw_beh_mat = raw_beh['ALLBEHmatrix']
