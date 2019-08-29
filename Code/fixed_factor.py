import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import scipy as sci

from functions import data
from functions import settings as sett 
from scipy.optimize import linear_sum_assignment

from functions import TCA as t
from sklearn.metrics import r2_score


torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load parameters 
param = sett.params()
paths = sett.paths()
ar = sett.arguments()

args = ar.get_arguments()
fixed_selection = ar.get_fixed_args()

def paring_r2(X, Y):
		"""Perform linear sum assignment based on R2 scores
		
		Parameters
		----------
		X : array
			Factor to compare
		Y: array
			Factor to compare

		Returns
		-------
		array
			Array rearranged
		array
			Array rearranged
		"""
		paring_matrix = np.empty(shape=(X.shape[1], X.shape[1]))
		
		for i, x in enumerate(X.T):
			for j, y in enumerate(Y.T):
				paring_matrix[i][j] = r2_score(x, y)

		row_ind, col_ind = linear_sum_assignment(paring_matrix)

		return X.T[row_ind].T, Y.T[col_ind].T

def load_factors(method):
	"""Load factors in order to generate global time factor
	
	Returns
	-------
	list
		List of factors
	"""
	factors = []
	for a in fixed_selection['Animal']:
		animal = param.animal_list[a]
		meta_df, roi_tensor, acti, norm_acti, smoothed_acti = data.load_processed_data_all(animal)
		for e in fixed_selection['Experiment Class']:
				days = set(meta_df.loc[meta_df['Experiment Class'] == e, 'Day'].tolist())
				print(days)
				for day in days:
					current_fixed_selection = {
					'Experiment Class': [e],
					'Day': [day]
					}
					print(current_fixed_selection)
					name = ''.join(['_' + k + '-' + str(current_fixed_selection[k]) for k in current_fixed_selection if not current_fixed_selection[k] == None])[1:]

					path = os.path.join(paths.path2Output, animal, method, args.init, str(args.rank), name)
					
					if os.path.exists(path):
						factor = np.load(os.path.join(path, 'factors' + name + '_00.npy'))
					else:
						os.makedirs(path)

						_, acti_select, norm_acti_select, smoothed_acti_select = data.select_data(meta_df, acti, norm_acti, smoothed_acti, current_fixed_selection)

						model = t.TCA(function=method, rank=args.rank, init=args.init, verbose=args.verbose, roi_tensor=roi_tensor)
						factor = model.fit(torch.tensor(norm_acti_select))
						
						np.save(os.path.join(path, 'factors{}_00.npy'.format(name)), factor)


					#Compute norms along columns for each factor matrix
					norms = [sci.linalg.norm(f, axis=0) for f in factor]

					# Multiply norms across all modes
					lam = sci.multiply.reduce(norms) ** (1/3)

					# Update factors
					factor = [f * (lam / fn) for f, fn in zip(factor, norms)]
					
					factors.append(factor)
	return factors

def rearrange_factors(factors):
	"""Rearrange all loaded factors to calculate mean

	Parameters
	----------
	factors : list
		List of factors to rearrange

	Returns
	-------
	list
		List of rearranged factors for later plotting
	list
		List of mean rearranged factor use for computation
	"""
	rearranged_factors = []
	for i, f in enumerate(factors):
		if i == 0:
			f1 = f[1]
			rearranged_factors.append(f1) 
		else:
			f1, f2 = paring_r2(f1, f[1])
			rearranged_factors.append(f2)
			rearranged_factors[0] = f1

	mean_factor = np.mean(np.array(rearranged_factors), axis=0)


	return rearranged_factors, mean_factor

def get_fixed_factor(method):
		"""Return an average time factor to fit TCA with inter-animal information

		Returns
		-------
		torch.Tensor
			Tensor of factors that will be used to fix the TCA
		"""
		factors = load_factors(method)
		rearranged_factors, mean_factor = rearrange_factors(factors)
		
		if not args.name:
			name = ''.join(['_' + k + '-' + str(fixed_selection[k]) for k in fixed_selection if not fixed_selection[k] == None])[1:]
		else: 
			name = args.name

		np.save(os.path.join(paths.path2fixed_factors, name), mean_factor)


		return torch.tensor(mean_factor)

if __name__ == '__main__':
	get_fixed_factor(args.function)