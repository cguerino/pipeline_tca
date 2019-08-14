import os
import numpy as np
import pandas as pd
from hdf5storage import loadmat
import pickle

from functions import settings as sett 
params = sett.params()
paths = sett.paths()


def load_data(animal, selection, verbose=False):
	""" Load raw data for a given animal depending on the selection made through the CLI. 
	
	Arguments
	---------
	animal : string
		Name of the animal
	selection : dict
		Selection preferences passed through the CLI. {'Selection':value}
	verbose : bool, optional
		Verbose mode
	
	Returns
	-------
	DataFrame
		Metadata for each selected trial
	array
		Mask of ROIsroi_tensor {array} -- mask of ROIs
	array
		Activation data depending on ROI, time and trial
	array
		Baseline fluorescence
	list
		List of trials matching with selection criteria
	"""
	path_raw = os.path.join(paths.path2Data, animal, 'Raw Data')
	
	raw_paths = [os.path.join(path_raw, file) for file in ['ROIBinmaps_{0}.mat'.format(animal), 'ALLDFF.mat', 'ALLBEHmatrix.mat', 'ALLf0.mat']]

	meta_df = pd.read_csv(os.path.join(path_raw, 'meta_df.csv'), index_col=0)
	
	# Load matrices as dictionnary {name: matrix_values}
	roi_file, raw_mat, raw_beh, f0 = list(map(loadmat, raw_paths))
	
	# Indexing data of interest
	roi_tensor = roi_file['ROIBinmaps']

	# Remove the first second of acquisition
	acti = raw_mat['ALLDFF'][:,15:,:]
	
	f0 = f0['F0ALL']

	if verbose: 
		N, T, K = acti.shape
		print('Data contain activity of', N, 'ROI, on a time scale of', T/15, 'seconds, during', K, 'trials')

	return meta_df, roi_tensor, acti, f0

def save_data(interpolated_acti, norm_acti, smoothed_acti, flag_roi, trials_to_drop, roi_tensor, meta_df, animal, name, arguments, selection):
	""" Save preprocessed data in order to save time for different computations. 
	
	Arguments
	---------
	interpolated_acti : array 
		Corrected activation tensor, all NaNs removed
	norm_acti : array 
		Normalized activity data dependnig on ROI, time and trial
	smoothed_acti : array 
		Smoothed and normalized activity data dependnig on ROI, time and trial
	flag_roi : list 
		Roi removed beacause of NaNs
	trials_to_drop : list 
		Trials to drop because they didn't match selection criteria
	roi_tensor : array -
		Mask of ROIs
	meta_df : DataFrame 
		Metadata for each selected trial
	animal : string 
		Name of the animal
	name : string 
		Random name of the current running simulation
	arguments : dict 
		Simulation configuration. {'Parameter': value}
	selection : dict 
		Selection preferences passed through the CLI. {'Selection':value}
	"""

	configuration = pd.concat([pd.DataFrame(arguments, index=[0]), pd.DataFrame(selection, index=[0])], axis=1)
	
	path = os.path.join(paths.path2Output, animal, str(arguments['Function']), 
						str(arguments['Init']), str(arguments['Rank']), name)
	try:
	    os.makedirs(path)
	except:
	    FileExistsError
	
	configuration.to_csv(os.path.join(path, 'configuration.csv'))
	np.save(os.path.join(path, 'acti'), interpolated_acti)
	np.save(os.path.join(path, 'norm_acti'), norm_acti)
	np.save(os.path.join(path, 'smoothed_acti'), smoothed_acti)
	np.save(os.path.join(path,'flag_roi'), flag_roi)
	np.save(os.path.join(path,'trials_to_drop'), trials_to_drop)
	np.save(os.path.join(path,'roi_tensor'), roi_tensor)
	meta_df.to_csv(os.path.join(path, 'meta_df.csv'))

def save_data_all(interpolated_acti, norm_acti, smoothed_acti, flag_roi, roi_tensor, meta_df, animal, arguments):
	""" Save preprocessed data in order to save time for different computations. 
	Arguments
	---------
	interpolated_acti : array 
		Corrected activation tensor, all NaNs removed
	norm_acti : array 
		Normalized activity data dependnig on ROI, time and trial
	smoothed_acti : array 
		Smoothed and normalized activity data dependnig on ROI, time and trial
	flag_roi : list 
		Roi removed beacause of NaNs
	roi_tensor : array
		Mask of ROIs
	meta_df : DataFrame 
		Metadata for each selected trial
	animal : string 
		Name of the animal
	arguments : dict 
		Simulation configuration. {'Parameter': value}
	"""

	configuration = pd.DataFrame(arguments, index=[0])
	
	path = os.path.join(paths.path2Output, animal, 'Preprocessed Data')
	try:
	    os.makedirs(path)
	except:
	    FileExistsError
	
	configuration.to_csv(os.path.join(path, 'configuration.csv'))
	np.save(os.path.join(path, 'acti'), interpolated_acti)
	np.save(os.path.join(path, 'norm_acti'), norm_acti)
	np.save(os.path.join(path, 'smoothed_acti'), smoothed_acti)
	np.save(os.path.join(path,'flag_roi'), flag_roi)
	np.save(os.path.join(path,'roi_tensor'), roi_tensor)
	meta_df.to_csv(os.path.join(path, 'meta_df.csv'))

def load_processed_data(animal, name, arguments):
	""" Load preprocessed data
	Arguments
	---------
	animal : string 
		Name of the animal
	name : string 
		Random name of the current running simulation
	arguments : dict -
		Simulation configuration. {'Parameter': value}

	Returns
	-------
	meta_df : DataFrame
		Metadata for each selected trial
	roi_tensor : array 
		Mask of ROIs
	acti : array 
		Activity data dependnig on ROI, time and trial
	norm_acti : array 
		Normalized activity data dependnig on ROI, time and trial
	smoothed_acti : array 
		Smoothed and normalized activity data dependnig on ROI, time and trial	
	"""
	path = os.path.join(paths.path2Output, animal, str(arguments['Function']), 
						str(arguments['Init']), str(arguments['Rank']), name)
	
	meta_df = pd.read_csv(os.path.join(path,'meta_df.csv'))
	roi_tensor = np.load(os.path.join(path,'roi_tensor.npy'))
	acti = np.load(os.path.join(path,'acti.npy'))
	norm_acti = np.load(os.path.join(path,'norm_acti.npy'))
	smoothed_acti = np.load(os.path.join(path,'smoothed_acti.npy'))


	return meta_df, roi_tensor, acti, norm_acti, smoothed_acti

def load_processed_data_all(animal):
	""" Load preprocessed data
	Arguments:
	animal : string 
		Name of the animal
	
	Returns
	-------
	meta_df : DataFrame
		Metadata for each selected trial
	roi_tensor : array 
		Mask of ROIs
	acti : array 
		Activity data dependnig on ROI, time and trial
	norm_acti : array 
		Normalized activity data dependnig on ROI, time and trial
	smoothed_acti : array 
		Smoothed and normalized activity data dependnig on ROI, time and tria	
	"""
	path = os.path.join(paths.path2Output, animal, 'Preprocessed Data')
	
	meta_df = pd.read_csv(os.path.join(path,'meta_df.csv'))
	roi_tensor = np.load(os.path.join(path,'roi_tensor.npy'))
	acti = np.load(os.path.join(path,'acti.npy'))
	norm_acti = np.load(os.path.join(path,'norm_acti.npy'))
	smoothed_acti = np.load(os.path.join(path,'smoothed_acti.npy'))

	return meta_df, roi_tensor, acti, norm_acti, smoothed_acti

def select_data(meta_df, acti, norm_acti, smoothed_acti, selection):
	for param in selection:
		if selection[param] != None:
			meta_df = meta_df[meta_df[param].isin(selection[param])]

	trials_of_interest = meta_df.index.tolist()

	if not trials_of_interest:
		raise ValueError('Selection is too restrictive, please select at least 1 trial')

	acti = acti[:,:,trials_of_interest]
	norm_acti = norm_acti[:, :, trials_of_interest]
	smoothed_acti = smoothed_acti[:, :, trials_of_interest]
	meta_df = meta_df.reset_index()

	return meta_df, acti, norm_acti, smoothed_acti

def save_results(factors, rec_errors, scores_odor, scores_rew, name, path):
	""" Save results of the TCA and random forests

	Arguments
	---------
	factors : array 
			Neuron, time and trial factors from TCA
	rec_errors : list 
			Reconstruction error during iterations of the TCA
	scores_odor : float 
			Prediction score of the odor presented
	scores_rew : float 
			Prediction score of the animal behavior
	path : string 
			Path to save data
	name : string 
			Random name of the current running simulation
	"""
	i = 0
	while os.path.exists(os.path.join(path, 'factors{}_{:02d}.npy').format(name, i)):
		i += 1
	
	np.save(os.path.join(path, 'factors{}_{:02d}.npy'.format(name, i)), factors)
	np.save(os.path.join(path, 'errors{}_{:02d}.npy'.format(name, i)), rec_errors)
	np.save(os.path.join(path, 'scores_odor{}_{:02d}.npy'.format(name, i)), scores_odor)
	np.save(os.path.join(path, 'scores_rew{}_{:02d}.npy'.format(name, i)), scores_rew)

def check_number_files(path):
	i = 0
	while os.path.exists(path):
		i += 1

	return i