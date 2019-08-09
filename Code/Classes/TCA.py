import os 
import numpy as np 
import tensorly as tl

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt 
class TCA:
	def __init__(self, function='non_negative_parafac', rank=6, init='random', max_iteration=10000, verbose=False, random_state=None, time_factor=None):
		self.rank = rank
		self.function = function
		self.init = init
		self.max_iteration = max_iteration
		self.verbose = verbose
		self.random_state = random_state

		self.dimension = 3
		self.epsilon = 1e-12
		self.rec_errors = []
		self.init_error = None
		self.converging_steps = None
		self.final_error = None

		self.factors = None
		self.time_factor = time_factor
		self.nb_estim = None
	def __add__(self):
		# Room to improvments of pipelines
		pass

	def _initialize_factors(self, tensor, svd='numpy_svd', non_negative=False, custom=None):
		rng = tl.random.check_random_state(self.random_state)
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
	
	def _get_error(self, iteration, tensor, factors, tol, verbose):
		norm_tensor = tl.norm(tensor, 2)
		if iteration % 25 == 0 and iteration > 1:
			rec_error = tl.norm(tensor - tl.kruskal_tensor.kruskal_to_tensor(factors), 2) / norm_tensor
			self.rec_errors.append(rec_error)
		
		if iteration % 25 == 1 and iteration > 1:
			rec_error = tl.norm(tensor - tl.kruskal_tensor.kruskal_to_tensor(factors), 2) / norm_tensor
			self.rec_errors.append(rec_error)
			print('reconstruction error={}, variation={}.'.format(
					self.rec_errors[-1], self.rec_errors[-2] - self.rec_errors[-1]))

			if abs(self.rec_errors[-2] - self.rec_errors[-1]) < tol:
				if verbose:
					print('converged in {} iterations.'.format(iteration))
					print('R2 scores : {}'.format(tca.compute_r2_score(tensor, kruskal_to_tensor(factors))))
				return True
			else:
				return False
	
	def _factor_non_negative(self, tensor, factors, mode):
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
	
	def _factor(self, tensor, factors, mode, pseudo_inverse):
		for i, factor in enumerate(factors):
			if i != mode:
				pseudo_inverse = pseudo_inverse*tl.dot(tl.transpose(factor), factor)
		factor = tl.dot(tl.base.unfold(tensor, mode), tl.tenalg.khatri_rao(factors, skip_matrix=mode))
		factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(factor)))
		
		return factor
	
	def _parafac(self, tensor, tol=1e-10):
		factors = self._initialize_factors(tensor, self.rank, self.init)
		pseudo_inverse = tl.tensor(np.ones((self.rank, self.rank)), **tl.context(tensor))
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				factors[mode] = self._factor(tensor, factors, mode, pseudo_inverse)
			
			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self._get_error(iteration, tensor, factors, tol, self.verbose):
					break
			return factors
	
	def _non_negative_parafac(self, tensor, tol=1e-7):
		factors = self._initialize_factors(tensor, non_negative=True)
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				factors[mode] = factors[mode]* self._factor_non_negative(tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self._get_error(iteration, tensor, factors, tol, self.verbose):
					break

		return factors
	
	def _custom_parafac(self, tensor, neg_fac=0, tol=1e-7):
		factors = self._initialize_factors(tensor, self.rank, self.init, custom=neg_fac)
		pseudo_inverse = tl.tensor(np.ones((self.rank, self.rank)), **tl.context(tensor))
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode == neg_fac:
					factors[mode] = self._factor(self, tensor, factors, mode, pseudo_inverse)
				else:
					factors[mode] = factors[mode] * self._factor_non_negative(tensor, factors, mode)
			
			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self._get_error(iteration, tensor, factors, tol, self.verbose):
					break
	
		return factors
	
	def _non_negative_fixed_parafac(self, tensor, time_factor):
		factors = self._initialize_factors(tensor, self.rank, self.init)
		factors[1] = time_factor

		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode != 1:
					factors[mode] = factors[mode] * self._factor_non_negative(self, tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self._get_error(iteration, tensor, factors, tol, self.verbose):
					break
		
		return factors
	
	def _fixed_parafac(self, tensor, time_factor):
		factors = self._initialize_factors(tensor, self.rank, self.init)
		factors[1] = time_factor
		for iteration in range(self.max_iteration):
			for mode in range(self.dimension):
				if mode != 1:
					factors[mode] = factors[mode] * self._factor(self, tensor, factors, mode)

			if (iteration % 25 == 0 or iteration % 25 == 1) and iteration > 1:
				if self._get_error(iteration, tensor, factors, tol, self.verbose):
					break
		
		return factors

	def _make_map(slef, roi_tensor, neuron_factor):
		"""Compute an image of the field of view with ROIs having different intensities
			
		Arguments:
			roi_tensor {boolean 3d array} -- 3-dimensional array of shape (512, 512, N) 
			where the slice (:,:,n) is a boolean mask for ROI n
			neuron_factor {list} -- list of length N with neuron factors of a component
			extracted from TCA
		
		Returns:
			2d array - image -- (512, 512) array
		"""
		
		roi_map = np.zeros([512, 512])
		for n in range(roi_tensor.shape[2]):
			roi_map += neuron_factor[n] * roi_tensor[:, :, n]
			
		return roi_map

	def error(self):
		return self.rec_errors

	def detailed_error(self):
		return self.rec_errors, self.init_error, self.converging_steps, self.final_error

	def fit(self, tensor):
		if self.function == 'custom_parafac':
			factors = self._custom_parafac(tensor, neg_fac=0, tol=1e-7)
		
		if self.function == 'parafac':
			factors = self._parafac(tensor, tol=1e-10)
		
		if self.function == 'non_negative_parafac':
			factors = self._non_negative_parafac(tensor)
		
		if self.function == 'fixed_parafac':
			factors = self._fixed_parafac(tensor, self.time_factor)
		
		if self.function == 'non_negative_fixed_parafac':
			factors = self._non_negative_fixed_parafac(tensor, self.time_factor)
		
		# Bring back factors array to cpu memory and convert it to numpy array
		factors = [f.cpu().numpy() for f in factors]
		self.factors = factors

		return self.factors

	def predict(self, meta_df, nb_estim=50):
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

	def important_features_map(self, roi_tensor, path_fig):
		feat_imp_odor = self.clf_odor.feature_importances_
		feat_imp_rew = self.clf_rew.feature_importances_ 
		
		for i, f in enumerate([feat_imp_odor, feat_imp_rew]):
			for r in range(self.factors[0].shape[1]):
				roi_map = self._make_map(roi_tensor, self.factors[0][:, r])
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

				if i:
					spe, i = 'rew', 0
					while os.path.exists(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i))):
						i += 1
					plt.savefig(os.path.join(path_fig, 'Maps', 'Importance_map_{}_{}_{:02d}.png'.format(spe, f[r],  i)))

	def important_features_time(self, roi_tensor, path_fig, spe=None):
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
				
				if i:
					spe, i = 'rew', 0
					while os.path.exists(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i))):
						i += 1
				else:
					plt.savefig(os.path.join(path_fig, 'Time', 'Importance_time_{}_{}_{:02d}.png'.format(spe, f[r], i)))