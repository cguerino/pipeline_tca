import os 
import heapq
import torch
import numpy as np 
import tensorly as tl
import tensorly.random

from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from functions import data
import matplotlib.pyplot as plt 
from functions import settings as sett 
from fixed_factor import get_fixed_factor

paths = sett.paths()
param = sett.params()
import scipy as sci

class TCA:
	"""
	Model for performing TCA optimization and post-fitting analysis

	Attributes
	----------
	rank : int
		Number of component in the TCA
	function : str
		Type of TCA algorithm 
	init : str
		Type of initialization to use for TCA
	max_iteration : int
		Maximum iterations allowed to reah convergence
	verbose : bool
		Verbose mode
	random_state : int
		Random State seed
	dimension : int
		Number of factors in the TCA
	epsilon : float
		Smallest number allowed in non-negative TCAs
	rec_errors : list
		Record of errors of TCA optimization
	init_error : float
		Initial error of the TCA
	converging_steps : int
		Number of steps that were necessary to reach convergence
	final_error : float
		Error at the end of the TCA
	factors : list
		List of TCA factors
	nb_estim : int
		Number of decision trees estimator in the random forest
	feat_odor : list
		List of indexes of best predictive ROI for odor
	feat_rew : list
		List of best predictive ROI for rew
	nb_feat : int
		Number of best predictive ROI kept

	Constructors
	------------
	__init__(self, function='non_negative_parafac', rank=6, init='random', max_iteration=10000, verbose=False, random_state=None, time_factor=None)
		Initialize object and attributes
	__add__(self, other)
		For later use of pipelines

	Methods
	-------
	error()
		Get record of errors during TCA optimization
	detailed_error()
		Get detailed information about error optimization and TCA algorith performance
	fit(tensor)
		Fit the model following a given TCA method
	predict(meta_df, nb_estim=50)
		Predict odor and reward metadata using a random forest classifier
	important_features_map(path_fig)
		Generate plots of TCA neuron factor components according to their predictive performance
	important_features_time(path_fig)
		Generate plots of TCA time factor components according to their predictive performance
	best_predictive_rois(self, path_fig, nb_feat=15)
		Extract best predictive ROIs and plot them on a map
	factorplot(self, meta_df, animal, name, path, balance=True, color='k', shaded=None, order=False)
		Display the factors extracted with TCA
	"""
	def __init__(self, function='non_negative_parafac', rank=6, init='random',  max_iteration=10000, verbose=False, random_state=None, roi_tensor=None, ff=None):
		"""Constructor at initialization

		Parameters
		----------
		function : str, optional
			Type of TCA algorithm (default is 'non_negative_parafac')
		rank : int, optional
			Number of component in the TCA (default is 6)
		init : str, optional
			Type of initialization to use for TCA (default is 'random')
		max_iteration: int, optional
			Maximum iterations allowed to reach convergence (default is 10_000)
		verbose : bool, optional
			Verbose mode (default is False)
		random_state : int, optional
			Random state seed
		roi_tensor : array
			3-dimensional array of shape (512, 512, N) where the slice (:,:,n) is a boolean mask for ROI n
		ff : str
			Fixed factor filename
		"""
		self.rank = rank
		self.function = function
		self.init = init
		self.max_iteration = max_iteration
		self.verbose = verbose
		self.random_state = random_state
		self.roi_tensor = roi_tensor
		self.ff = ff
		self.dimension = 3
		self.epsilon = 1e-12
		self.rec_errors = []
		self.init_error = None
		self.converging_steps = None
		self.final_error = None
		self.factors = None
		self.nb_estim = None
		self.feat_odor = None
		self.feat_rew = None
		self.nb_feat = None
	def __add__(self, other):
		# For later use of pipelines
		pass


	def __initialize_factors(self, tensor, svd='numpy_svd', non_negative=False, custom=None):
		"""Initialize random or SVD-guided factors for TCA depending on TCA type

		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		svd : str, optional
			Type of SVD algorithm to use (default is numpy_svd)
		non_negative : bool, optional
			A flag used to specify if factors generated must be strictyl positive (default is False)
		custom : int, optional 
			A flag used to specify which factor should be strictly positive for 'custom parafac' (default is None)

		Raises
		------
		ValueError
			If svd does not contain a valid SVD algorithm reference
			If self.init variable does not contain a valid intialization method

		Returns
		-------
		list
			List of initialized tensors
		"""
		rng = tensorly.random.check_random_state(self.random_state)
		if self.init == 'random':
			if custom:
				factors = [tl.tensor(rng.random_sample((tensor.shape[i], self.rank))*2 - 1, **tl.context(tensor)) for i in range(self.dimension)]
				factors = [f if int(i) == int(custom) else tl.abs(f) for i, f in enumerate(factors)]
			
			elif non_negative:
				factors = [tl.tensor(rng.random_sample((tensor.shape[i], self.rank)), **tl.context(tensor)) for i in range(self.dimension)]
				factors = [tl.abs(f) for f in factors] # See if this line is useful depending on random function used
			
			else:
				factors = [tl.tensor(rng.random_sample((tensor.shape[i], self.rank))*2 - 1, **tl.context(tensor)) for i in range(self.dimension)]

			return factors

		elif self.init == 'svd':
			try:
				svd_fun = tl.SVD_FUNS[svd]
			except KeyError:
				message = 'Got svd={}. However, for the current backend ({}), the possible choices are {}'.format(
						svd, tl.get_backend(), tl.SVD_FUNS)
				raise ValueError(message)

			factors = []
			for mode in range(tl.ndim(tensor)):
				U, *_ = svd_fun(unfold(tensor, mode), n_eigenvecs=rank)

				if tensor.shape[mode] < rank:
					random_part = tl.tensor(rng.random_sample((U.shape[0], rank - tl.shape(tensor)[mode])), **tl.context(tensor))
					
					U = tl.concatenate([U, random_part], axis=1)
				
				if non_negative or custom == mode:
					factors.append(tl.abs(U[:, :rank]))
				else:
					factors.append(U[:, :rank])
			
			return factors
		else:
			raise ValueError('Initialization method "{}" not recognized'.format(self.init))
	def __compute_r2_score(self, tensor, factors):
		"""Compute R2 score between all time factor curves
		
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		factors : list
			List of tensors, each one containing a factor

		Raises
		------
		TypeError
			If factors is alrerady in CPU memory

		Returns
		-------
		float
			Mean of R2 scores
		"""
		N, T, K = tensor.shape

		tensor_traces = [tensor[n, :, k] for n in range(N) for k in range(K)]
		tensor_traces = [t.cpu() for t in tensor_traces]
		factors_traces = [factors[n, :, k] for n in range(N) for k in range(K)]
		try:
			factors_traces = [f.cpu() for f in factors_traces]
		except:
			TypeError

		scores = [r2_score(tensor_traces[i], factors_traces[i]) for i in range(len(tensor_traces))]

		return np.mean(scores)
	
	def __get_error(self, iteration, tensor, factors, tol=1e-7, verbose=False):
		"""Compute error of the TCA and check if convergence is reached each 25 iterations

		Parameters
		----------
		iteration : int
			Current number of iterations of the TCA optimization algorithm
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		factors : list
			List of tensors, each one containing a factor
		tol : float, optional 
			Threshold for convergence (default is 1e-7)
		verbose : bool, optional
			Verbose mode
		
		Returns
		-------
		bool
			A boolean that states if convergence is reached or not
		"""
		norm_tensor = tl.norm(tensor, 2)
		if iteration % 25 == 0 and iteration > 1:
			rec_error = tl.norm(tensor - tl.kruskal_tensor.kruskal_to_tensor(factors), 2) / norm_tensor
			self.rec_errors.append(rec_error)
		
		if iteration % 25 == 1 and iteration > 1:
			rec_error = tl.norm(tensor - tl.kruskal_tensor.kruskal_to_tensor(factors), 2) / norm_tensor
			self.rec_errors.append(rec_error)
			if verbose: print('reconstruction error={}, variation={}.'.format(
					self.rec_errors[-1], self.rec_errors[-2] - self.rec_errors[-1]))

			if abs(self.rec_errors[-2] - self.rec_errors[-1]) < tol:
				if verbose:
					print('converged in {} iterations.'.format(iteration))
					print('R2 scores : {}'.format(self.__compute_r2_score(tensor, tl.kruskal_tensor.kruskal_to_tensor(factors))))
				return True
			else:
				return False
	
	def __factor_non_negative(self, tensor, factors, mode):
		"""Compute a non-negative factor optimization for TCA

		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		factors : list
			List of tensors, each one containing a factor
		mode : int
			Index of the factor to optimize

		Returns
		-------
		float
			Number to which multiply the factor to for optimization

		"""
		sub_indices = [i for i in range(self.dimension) if i != mode]
		for i, e in enumerate(sub_indices):
			if i:
				accum = accum*tl.dot(tl.transpose(factors[e]), factors[e])
			else:
				accum = tl.dot(tl.transpose(factors[e]), factors[e])

		numerator = tl.dot(tl.base.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
		numerator = tl.clip(numerator, a_min=self.epsilon, a_max=None)
		denominator = tl.dot(factors[mode], accum)
		denominator = tl.clip(denominator, a_min=self.epsilon, a_max=None)
		
		return (numerator / denominator)
	
	def __factor(self, tensor, factors, mode, pseudo_inverse):
		"""Compute a factor optimization for TCA

		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		factors : list
			List of tensors, each one containing a factor
		mode : int
			Index of the factor to optimize
		pseudo_inverse : torch.Tensor
			Pseudo inverse matrix of the current factor

		Returns
		-------
		torch.Tensor
			Optimized factor

		"""
		for i, factor in enumerate(factors):
			if i != mode:
				pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
		factor = tl.dot(tl.base.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
		factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
		
		return factor
	
	def __parafac(self, tensor, tol=1e-10):
		"""Regular PARAFAC algorithm
			
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		tol : float, optional
			Threshold for convergence (default is 1e-10)

		Returns
		-------
		list
			List of optimized factors
		"""
		factors = self.__initialize_factors(tensor, self.rank, self.init)
		pseudo_inverse = tl.tensor(np.ones((self.rank, self.rank)), **tl.context(tensor))
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				factors[mode] = self.__factor(tensor, factors, mode, pseudo_inverse)
			
			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self.__get_error(iteration, tensor, factors, tol, self.verbose):
					break
			return factors
	
	def __non_negative_parafac(self, tensor, tol=1e-7):
		"""Non-negative PARAFAC algorithm
			
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		tol : float, optional
			Threshold for convergence (default is 1e-7)

		Returns
		-------
		list
			List of optimized factors
		"""
		factors = self.__initialize_factors(tensor, non_negative=True)
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				factors[mode] = factors[mode]* self.__factor_non_negative(tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self.__get_error(iteration, tensor, factors, tol, self.verbose):
					break

		return factors
	
	def __custom_parafac(self, tensor, neg_fac=0, tol=1e-7):
		"""Customized PARAFAC algorithm
			
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		neg_fac : int, optional
			Index of the factor which is allowed to be negative (default is 0)
		tol : float, optional
			Threshold for convergence (default is 1e-7)

		Returns
		-------
		list
			List of optimized factors
		"""
		factors = self.__initialize_factors(tensor, self.rank, self.init, custom=neg_fac)
		pseudo_inverse = tl.tensor(np.ones((self.rank, self.rank)), **tl.context(tensor))
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode == neg_fac:
					factors[mode] = self.__factor(self, tensor, factors, mode, pseudo_inverse)
				else:
					factors[mode] = factors[mode] * self.__factor_non_negative(tensor, factors, mode)
			
			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self.__get_error(iteration, tensor, factors, tol, self.verbose):
					break
	
		return factors
	
	def __non_negative_fixed_parafac(self, tensor, tol=1e-7):
		"""Non-negative PARAFAC algorithm with fixed time factor
			
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		tol : float, optional
			Threshold for convergence (default is 1e-7)

		Returns
		-------
		list
			List of optimized factors
		"""
		if not self.ff:
			time_factor = get_fixed_factor('non_negative_parafac')
			print('test')
		else:
			time_factor = torch.tensor(np.load(os.path.join(paths.path2fixed_factors, self.ff)))

		
		factors = self.__initialize_factors(tensor, self.rank, self.init)
		factors[1] = time_factor

		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode != 1:
					factors[mode] = factors[mode] * self.__factor_non_negative(tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self.__get_error(iteration, tensor, factors, tol, self.verbose):
					break
		
		return factors
	
	def __fixed_parafac(self, tensor, tol=1e-7):
		"""PARAFAC algorithm with fixed time factor
			
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K
		time_factor : list, optional
			List of factors from which to take time factor for fixing
		tol : float, optional
			Threshold for convergence (default is 1e-7)

		Returns
		-------
		list
			List of optimized factors
		"""
		time_factor = get_fixed_factor('parafac')
		factors = self.__initialize_factors(tensor, self.rank, self.init)
		factors[1] = time_factor
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode != 1:
					factors[mode] = factors[mode] * self.__factor(tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self.__get_error(iteration, tensor, factors, tol, self.verbose):
					break
		
		return factors

	def __make_map(self, neuron_factor):
		"""Compute an image of the field of view with ROIs having different intensities
			
		Parameters
		----------
		neuron_factor : list} 
			list of length N with neuron factors of a component extracted from TCA
		
		Returns
		-------
		array
			(512, 512) array image
		"""
		
		roi_map = np.zeros([512, 512])
		for n in range(self.roi_tensor.shape[2]):
			roi_map += neuron_factor[n] * self.roi_tensor[:, :, n]
			
		return roi_map

	def error(self):
		"""Get record of errors during TCA optimization

		Returns
		-------
		list
			Record of errors of the TCA optimization
		"""
		return self.rec_errors

	def detailed_error(self):
		"""Get detailed information about error optimization and TCA algorith performance
		
		Returns
		-------
		list
			Record of errors of the TCA optimization
		float
			Initial error of the TCA
		int
			Number of steps that was necessary for the TCA to converge
		float 
			Error of the TCA optimization at convergence
		"""
		return self.rec_errors, self.init_error, self.converging_steps, self.final_error

	def fit(self, tensor):
		"""Fit the model following a given TCA method
		
		Parameters
		----------
		tensor : torch.Tensor
			The tensor of activity of N neurons, T timepoints and K trials of shape N, T, K

		Returns
		-------
		list
			List of factors optimized
		"""
		if self.function == 'custom_parafac':
			factors = self.__custom_parafac(tensor, neg_fac=0, tol=1e-7)
		
		if self.function == 'parafac':
			factors = self.__parafac(tensor, tol=1e-10)
		
		if self.function == 'non_negative_parafac':
			factors = self.__non_negative_parafac(tensor)
		
		if self.function == 'fixed_parafac':
			factors = self.__fixed_parafac(tensor)
		
		if self.function == 'non_negative_fixed_parafac':
			factors = self.__non_negative_fixed_parafac(tensor)
		
		# Bring back factors array to cpu memory and convert it to numpy array
		factors = [f.cpu().numpy() for f in factors]
		self.factors = factors

		return self.factors

	def predict(self, meta_df, nb_estim=50):
		"""Predict odor and reward metadata using a random forest classifier

		Parameters
		----------
		meta_df : pandas.DataFrame
			Dataframe containing all the specifities about each trial
		nb_estim : int, optional
			Number of decision trees in the forest

		Returns
		-------
		float
			Prediction score for odor data
		float
			Prediction score for reward data
		RandomForestClassifier
			Model used for odor prediction
		RandomForestClassifier
			Model used for reward prediction
		"""
		if not self.factors:
			raise TypeError('Please fit the model before prediction')
		self.nb_estim = nb_estim

		X, y_odor, y_rew = shuffle(self.factors[2], meta_df['Odor'].tolist(), meta_df['Reward'].tolist())
		self.clf_odor = RandomForestClassifier(n_estimators=self.nb_estim, max_depth=None, min_samples_split=2, max_features='sqrt', oob_score=True)
		self.clf_rew = RandomForestClassifier(n_estimators=self.nb_estim, max_depth=None, min_samples_split=2, max_features='sqrt', oob_score=True)
		X = StandardScaler().fit_transform(X)
		self.clf_odor.fit(X, y_odor)
		self.score_odor = self.clf_odor.oob_score_
		self.clf_rew.fit(X, y_rew)
		self.score_rew = self.clf_rew.oob_score_

		return self.score_odor, self.score_rew, self.clf_odor, self.clf_rew

	def important_features_map(self, path_fig):
		"""Generate plots of TCA neuron factor components according to their predictive performance

		Parameters
		----------
		path_fig : str
			Path where the figures should be saved
		"""
		feat_imp_odor = self.clf_odor.feature_importances_
		feat_imp_rew = self.clf_rew.feature_importances_ 
		
		for i, f in enumerate([feat_imp_odor, feat_imp_rew]):
			for r in range(self.factors[0].shape[1]):
				roi_map = self.__make_map(self.factors[0][:, r])
				if np.min(self.factors[0][:, r]) < 0:
					plt.imshow(roi_map, vmin=0, vmax=np.max(self.factors[0]), cmap='coolwarm')
				else:
					plt.imshow(roi_map, vmin=0, vmax=np.max(self.factors[0]), cmap='hot')

				plt.title('Importance : {}'.format(f[r]))
				
				try:
					os.makedirs(os.path.join(path_fig, 'Maps')) 
				except:
					FileExistsError				
				
				if not i:
					spe, i = 'odor', 0
					while os.path.exists(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i))):
						i += 1
					plt.savefig(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i)))
					plt.clf()

				if i:
					spe, i = 'rew', 0
					while os.path.exists(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i))):
						i += 1
					plt.savefig(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i)))
					plt.clf()

	def important_features_time(self, path_fig):
		"""Generate plots of TCA time factor components according to their predictive performance

		Parameters
		----------
		path_fig : str
			Path where the figures should be saved
		"""
		feat_imp_odor = self.clf_odor.feature_importances_
		feat_imp_rew = self.clf_rew.feature_importances_

		T = self.factors[1].shape[0]
		shaded = [7, 9]
		for i, f in enumerate([feat_imp_odor, feat_imp_rew]):
			for r in range(self.factors[1].shape[1]):
				## plot time factors as a lineplot
				plt.plot(np.arange(1, T+1), self.factors[1][:, r], color='k', linewidth=2)
				# arrange labels on x axis
				plt.locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
				# color the shaded region
				plt.fill_betweenx([np.min(self.factors[1]), np.max(self.factors[1])+.01], 15*shaded[0],
										  15*shaded[1], facecolor='red', alpha=0.5)

				plt.title('Importance : {}'.format(f[r]))

				try:
					os.makedirs(os.path.join(path_fig, 'Time'))
				except:
					FileExistsError
				
				if not i:
					spe, i = 'odor', 0
					while os.path.exists(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i))):
						i += 1
					plt.savefig(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i)))
					plt.clf()
				
				if i:
					spe, i = 'rew', 0
					while os.path.exists(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i))):
						i += 1
				else:
					plt.savefig(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i)))
					plt.clf()

	def best_predictive_rois(self, path_fig, nb_feat=15):
		"""Extract best predictive ROIs and plot them on a map

		Parameters
		----------
		path_fig : str
			Path where the figures should be saved
		nb_feat : int, optional
			Number of best ROIs plotted
		"""
		feat_odor = self.clf_odor.feature_importances_
		feat_rew = self.clf_rew.feature_importances_
		self.nb_feat = nb_feat
		for k, feat_ in enumerate([feat_odor, feat_rew]):
			max_feat_idx = np.argmax(feat_)
			rois = list(self.factors[0][:,max_feat_idx])
			best_rois = heapq.nlargest(nb_feat, enumerate(rois), key=lambda x: x[1])
			best_indices = [best_rois[i][0] for i in range(len(best_rois))]
			best_values = [best_rois[i][1] for i in range(len(best_rois))]

			reshaped_best_rois = [0]*len(self.factors[0][:, max_feat_idx])
			for i, j in zip(best_indices, best_values):
				reshaped_best_rois[i] = j

			roi_map = self.__make_map(reshaped_best_rois)
			if np.min(best_values) < 0:
				plt.imshow(roi_map, vmin=0, vmax=np.max(best_values), cmap='coolwarm')
			else:
				plt.imshow(roi_map, vmin=0, vmax=np.max(best_values), cmap='hot')

			plt.title('Importance : {}'.format(feat_[max_feat_idx]))

			try:
				os.makedirs(os.path.join(path_fig, 'Reduce_roi'))
			except:
				FileExistsError
			
			if not k:
				i = 0  
				while os.path.exists(os.path.join(path_fig, 'Reduce_roi', 'Importance_map_odor_{}_{:02d}.png'.format(feat_[max_feat_idx], i))):
					i += 1

				plt.savefig(os.path.join(path_fig, 'Reduce_roi', 'Importance_map_odor_{}_{:02d}.png'.format(feat_[max_feat_idx], i)))
				self.feat_odor = best_indices
				self.reshaped_best_rois_odor = reshaped_best_rois
			if k:
				i = 0
				while os.path.exists(os.path.join(path_fig, 'Reduce_roi', 'Importance_map_rew_{}_{:02d}.png'.format(feat_[max_feat_idx], i))):
					i += 1

				plt.savefig(os.path.join(path_fig, 'Reduce_roi', 'Importance_map_rew_{}_{:02d}.png'.format(feat_[max_feat_idx], i)))
				self.feat_rew = best_indices
				self.reshaped_best_rois_rew = reshaped_best_rois

		plt.clf()
		
		return self.feat_odor, self.feat_rew, self.reshaped_best_rois_odor, self.reshaped_best_rois_rew

	def get_fixed_factor(self):
		"""Return an average time factor to fit TCA with inter-animal information

		Returns
		-------
		torch.Tensor
			Tensor of factors that will be used to fix the TCA
		"""
		factors = self.__load_factors()
		rearranged_factors, mean_factor = self.__rearrange_factors(factors)

		return torch.tensor(mean_factor)

	def __give_order(self):
		"""Return the order in which components should be plotted
		
		Returns
		-------
		list
			List of indexes
		"""

		onset = []

		for r in range(1, self.rank):
			comp = self.factors[1][:, r]
			thres = 2 * np.std(comp) + comp[0]

			t = 0
			while comp[t] < thres:
				if t < 284:
					t += 1
				else:
					break
			onset.append(t)

		order = []
		ti = np.copy(onset)
		onset.sort()

		for t in onset:
			i = 0
			while t != ti[i]:
				i += 1
			order.append(i)

		return order

	def __ord_fact(self, order):
		"""Re-order factors given indices

		Arguments
		---------
		order : list
			List of indexes in order

		Returns
		-------
			List
				Factors ordered
		"""

		ord_factors = []

		for factor in factors:
			ord_factor = np.zeros_like(factor)
			ord_factor[:, 0] = factor[:, 0]
			for i, o in enumerate(order):
				ord_factor[:, i+1] = factor[:, o+1]
			ord_factors.append(ord_factor)

		return ord_factors
	
	def factorplot(self, meta_df, animal, name, path, balance=True, color='k', shaded=None, order=False):
		"""Display the factors extracted with TCA
		
		The TCA extracted factors are represented in 3 columns and as many rows as there are components.
		On the first column neuron factors are represented on the ROI map.
		On the second column temporal factors are represented.
		On the third column trial factors are represented.
		
		Arguments
		---------
		meta_df : pandas.DataFrame
			Dataframe containing all the specifities about each trial
		animal : string 
			Name of the animal
		name : string
			String to append to filename
		balance : bool, optional
			whether factors be normalized over modes and components (default : True)
		color : list
			Color code for each trial factor (default : 'k')
		shaded : list
			Interval in seconds to be shaded in temporal factor (default : None)
		order : bool
			Wether components should be ordered by activity onset (default : False)
		path : string
			Destination folder for saving (default : None)
		"""
			
		# whether or not factors columns should be normalized to unit norm
		if balance:
			# Compute norms along columns for each factor matrix
			# norms is a list of 3 arrays of length rank
			norms = [sci.linalg.norm(f, axis=0) for f in self.factors]

			# Multiply norms across all modes to have 1 norm per component
			# lam is a list of length rank
			lam = sci.multiply.reduce(norms) ** (1/3)

			# Update factors to normalize each columns to unit norm
			self.factors = [f * (lam / fn) for f, fn in zip(self.factors, norms)]
		
		# wheter or not components should be ordered by activity onset
		if order:
			self.factors = self.__ord_fact(self.factors, self.__give_order(self.factors))

		# rank is the number of components of TCA - as well as the number of columns in factor matrices
		rank = self.factors[0].shape[1]
		# T is the number of timeframes for each trial, usually 285
		T = self.factors[1].shape[0]
		# K is the number of trials
		K = self.factors[2].shape[0]


		limit_block = [[meta_df[meta_df['Block'] == i].index.tolist()[0], meta_df[meta_df['Block'] == i].index.tolist()[-1]] for i in set(meta_df['Block'].tolist())]
		limit_day = [meta_df[meta_df['Day'] == i].index.tolist()[-1] for i in set(meta_df['Day'].tolist())]
		learn_color = [meta_df['Performance Color'].iloc[x[0]] for x in limit_block]
		top = np.max(self.factors[2])

		# by default the shaded interval for trial factors
		# corresponds to the odor presentation
		if shaded is None:
			# interval in seconds, 1 second = 15 timeframes
			shaded = [7, 9]
		
		# initiate the plotting object
		fig, axarr = plt.subplots(rank, 3, sharex='col', figsize=(15, rank*3))

		# for each of the component r
		for r in range(rank):

			## plot neuron factors on the ROI map
			# generate the image with ROI binary tensor and neuron factors

			roi_map = self.__make_map(self.factors[0][:, r])
			# plot as an image, beware normalized colormap
			axarr[r, 0].imshow(roi_map, vmin=0, vmax=np.max(self.factors[0]), cmap='hot')

			## plot time factors as a lineplot
			axarr[r, 1].plot(np.arange(1, T+1), self.factors[1][:, r], color='k', linewidth=2)
			# arrange labels on x axis
			axarr[r, 1].locator_params(nbins=T//30, steps=[1, 3, 5, 10], min_n_ticks=T//30)
			# color the shaded region
			axarr[r, 1].fill_betweenx([np.min(self.factors[1]), np.max(self.factors[1])+.01], 15*shaded[0],
									  15*shaded[1], facecolor='red', alpha=0.5)

			## plot trial factors as a scatter plot
			axarr[r, 2].scatter(np.arange(1, K+1), self.factors[2][:, r], c=color)
			# arrange labels on x axis
			axarr[r, 2].locator_params(nbins=K//20, steps=[1, 2, 5, 10], min_n_ticks=K//20)
			
			# add information from behaviorgram if needed

			# iterate over blocks
			for i, block in enumerate(limit_block):
				# color a region over trial factors corresponding to a given block
				# the color denotes the learning score for the given block
				axarr[r, 2].fill_betweenx([1.05 * top, 1.25 * top], block[0], block[1], 
										  facecolor=learn_color[i], alpha=1)
			# iterate over days
			for limit in limit_day:
				# plot a black line for day shift between learning score colors
				axarr[r, 2].axvline(limit, 0.75, 1, linewidth=2, color='black')
			
			# for mode 1 and 2 (i.e temporal and trial factors)
			for i in [1, 2]:

				# format axes, remove spines for all components
				axarr[r, i].spines['top'].set_visible(False)
				axarr[r, i].spines['right'].set_visible(False)

				# remove xticks on all but bottom row, to keep legend on this row
				if r != rank-1:
					plt.setp(axarr[r, i].get_xticklabels(), visible=False)
			
			# remove axes, spines and labels for neuron factors
			axarr[r, 0].tick_params(axis='both', which='both', bottom=False, top=False,
									labelbottom=False, right=False, left=False, labelleft=False)

		# set titles for top row and legend for bottom row
		axarr[0, 0].set_title('Neuron Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
		axarr[0, 1].set_title('Temporal Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
		axarr[0, 2].set_title('Trial Factors', {'fontsize': 'x-large', 'fontweight' : 'roman'})
		
		# set label for bottom row trial factors
		axarr[rank-1, 0].set_xlabel('ROI map', {'fontsize': 'large', 'fontweight' : 'bold',
												'verticalalignment' : 'top'})

		# generate time index in seconds for temporal factors
		time_index = list(np.arange(0, T//15 + 1, 2))
		# insert another 0 to preserve length
		time_index.insert(0, 1)
		# set label for bottom row temporal factors
		axarr[rank-1, 1].set_xlabel('Time (s)', {'fontsize': 'large', 'fontweight' : 'bold'})
		# set ticks labels for bottom row temporal factors
		axarr[rank-1, 1].set_xticklabels(time_index)
		
		# generate trial index for trial factors
		trial_index = list(np.arange(0, K//20 + 2))
		# insert another 0 to preserve length
		trial_index.insert(0, 1)
		# set label for bottom row trial factors
		axarr[rank-1, 2].set_xlabel('Block', {'fontsize': 'large', 'fontweight' : 'bold'})
		# set ticks labels for bottom trial factors 
		axarr[rank-1, 2].set_xticklabels(trial_index)
		
		## link y-axes within columns
		# iterate over modes
		for i in range(3):
			# get amplitudes for each component
			yl = [a.get_ylim() for a in axarr[:, i]]
			# get maximum amplitudes, global minimum and maximum for factors
			y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
			# set same plotting intervals across components 
			_ = [a.set_ylim((y0, y1)) for a in axarr[:, i]]

		## format y-ticks
		# iterate over components
		for r in range(rank):
			# iterate over modes
			for i in range(3):
				# limit to two labels, minimum and maximum for y axis
				axarr[r, i].set_ylim(np.round(axarr[r, i].get_ylim(), 2))
				# set ticks accordingly
				axarr[r, i].set_yticks([0, np.round(axarr[r, i].get_ylim(), 2)[1]])

		# make so that plots are tightly presented
		plt.tight_layout()

		# display and save figure
		
		i = 0
		while os.path.exists(os.path.join(path, 'factorplot{}_{:02d}.png'.format(name, i))):
			i += 1

		plt.savefig(os.path.join(path, 'factorplot_{}_{:02d}.png'.format(name, i)))
		plt.clf()



