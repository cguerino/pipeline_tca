import os
import glob

import numpy as np 
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment

from functions import settings as sett
from functions import data 

import matplotlib.pyplot as plt 
from tqdm import tqdm 
import scipy as sci


paths = sett.paths()
param = sett.params()

def get_fixed_factor():
	factors = load_factors():
	rearranged_factors, mean_factor = rearrange_factors(factors)

	return mean_factor



def paring_r2(factors1, factors2):
	paring_matrix = np.empty(shape=(factors1.shape[1], factors1.shape[1]))
	
	for i, fac1 in enumerate(factors1.T):
		for j, fac2 in enumerate(factors2.T):
			paring_matrix[i][j] = r2_score(fac1, fac2)

	row_ind, col_ind = linear_sum_assignment(paring_matrix)

	return factors1.T[row_ind].T, factors2.T[col_ind].T

def load_factors():
	factors = []
	for a in range(12, 15):
		for e in [1, 3]:
			animal = param.animal_list[a]
			meta_df, roi_tensor, acti, norm_acti, smoothed_acti = data.load_processed_data_all(animal)
			days = set(meta_df.loc[meta_df['Experiment Class'] == e, 'Day'].tolist())
			for d in days:
				path = os.path.join(paths.path2Output, animal, 'non_negative_parafac', 'random', '6', 'Day-[{}]_Experiment Class-[{}]'.format(d, e),
								   'factorsDay-[{}]_Experiment Class-[{}]_00.npy'.format(d, e))
				print(path)
				factor = np.load(path)
			    #Compute norms along columns for each factor matrix
				norms = [sci.linalg.norm(f, axis=0) for f in factor]

				# Multiply norms across all modes
				lam = sci.multiply.reduce(norms) ** (1/3)

				# Update factors
				factor = [f * (lam / fn) for f, fn in zip(factor, norms)]
				
				factors.append(factor)
	return factors

def rearrange_factors(factors):
	rearranged_factors = []
	for i, f in tqdm(enumerate(factors)):
		if i == 0:
			f1 = f[1]
			rearranged_factors.append(f1) 
		else:
			f1, f2 = paring_r2(f1, f[1])
			rearranged_factors.append(f2)
			rearranged_factors[0] = f1

	mean_factor = np.mean(np.array(rearranged_factors), axis=0)


	return rearranged_factors, mean_f


def plot_fixed(rearranged_factors, mean_f):
	fig, axes = plt.subplots(6, len(factors))

	T = rearranged_factors[0].shape[0]
	shaded = [7, 9]
	for i, f in enumerate(rearranged_factors):
		print(f)
		print(f.shape)
		for r in range(f.shape[1]):
			## plot time factors as a lineplot
			axes[r, i].plot(np.arange(1, T+1), f[:, r], color='k', linewidth=2)
			# arrange labels on x axis
			axes[r, i].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
			# color the shaded region
			axes[r, i].fill_betweenx([np.min(f), np.max(f)+.01], 15*shaded[0],
									  15*shaded[1], facecolor='red', alpha=0.5)

			axes[r, i].set_yticks([])
			axes[r, i].set_xticks([])


	plt.savefig('test.png')
	plt.clf()
	print(np.array(rearranged_factors).shape)

	print(mean_f.shape)
	fig, axes = plt.subplots(6, 1)

	for r in range(mean_f.shape[1]):
		## plot time factors as a lineplot
		axes[r].plot(np.arange(1, T+1), mean_f[:, r], color='k', linewidth=2)
		# arrange labels on x axis
		axes[r].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
		# color the shaded region
		axes[r].fill_betweenx([np.min(mean_f), np.max(mean_f)+.01], 15*shaded[0],
								  15*shaded[1], facecolor='red', alpha=0.5)

		axes[r].set_yticks([])
		axes[r].set_xticks([])
	plt.savefig('test2.png')
	plt.show()