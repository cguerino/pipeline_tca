import os
import numpy as np
import pandas as pd 
from functions import data

from functions import settings as sett 
from munkres import Munkres
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 

from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

paths = sett.paths()


def kruskal_align(U, V, permute_U=False, permute_V=False):
    """Aligns two KTensors and returns a similarity score.

    Parameters
    ----------
    U : KTensor
        First kruskal tensor to align.
    V : KTensor
        Second kruskal tensor to align.
    permute_U : bool
        If True, modifies 'U' to align the KTensors (default is False).
    permute_V : bool
        If True, modifies 'V' to align the KTensors (default is False).

    Notes
    -----
    If both `permute_U` and `permute_V` are both set to True, then the
    factors are ordered from most to least similar. If only one is
    True then the factors on the modified KTensor are re-ordered to
    match the factors in the un-aligned KTensor.

    Returns
    -------
    similarity : float
        Similarity score between zero and one.
    """

    # Compute similarity matrices.
    unrm = [f / np.linalg.norm(f, axis=0) for f in U]
    vnrm = [f / np.linalg.norm(f, axis=0) for f in V]
    sim_matrices = [np.dot(u.T, v) for u, v in zip(unrm, vnrm)]
    cost = 1 - np.mean(np.abs(sim_matrices), axis=0)

    # Solve matching problem via Hungarian algorithm.
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # Compute mean factor similarity given the optimal matching.
    similarity = np.mean(1 - cost[prmU, prmV])

    # If U and V are of different ranks, identify unmatched factors.
    unmatched_U = list(set(range(U[0].shape[1])) - set(prmU))
    unmatched_V = list(set(range(V[0].shape[1])) - set(prmV))

    # If permuting both U and V, order factors from most to least similar.
    if permute_U and permute_V:
        idx = np.argsort(cost[prmU, prmV])

    # If permute_U is False, then order the factors such that the ordering
    # for U is unchanged.
    elif permute_V:
        idx = np.argsort(prmU)

    # If permute_V is False, then order the factors such that the ordering
    # for V is unchanged.
    elif permute_U:
        idx = np.argsort(prmV)

    # If permute_U and permute_V are both False, then we are done and can
    # simply return the similarity.
    else:
        return similarity

    # Re-order the factor permutations.
    prmU = [prmU[i] for i in idx]
    prmV = [prmV[i] for i in idx]

    # Permute the factors.
    if permute_U:
        U = [f[:, prmU] for f in U]
    if permute_V:
        V = [f[:, prmV] for f in V]

    # Flip the signs of factors.
    flips = np.sign([F[prmU, prmV] for F in sim_matrices])
    flips[0] *= np.prod(flips, axis=0)  # always flip an even number of factors

    if permute_U:
        for i, f in enumerate(flips):
            U[i] *= f

    elif permute_V:
        for i, f in enumerate(flips):
            V[i] *= f

    # Return the similarity score
    return similarity

def TSNE_trials(acti, meta_df, name, path, dataset='factors'):
	days = list(set(meta_df['Day'].tolist()))
	blocks = list(set(meta_df['Block'].tolist()))

	for day in days:
		trials_of_interest = meta_df[meta_df.Day == day].index.tolist()
		colors = meta_df.loc[trials_of_interest, 'Behavior Color'].tolist()
		
		if dataset == 'factors':
			X = factors[2][trials_of_interest, :]
		elif dataset == 'acti':
			X = acti[:, :, trials_of_interest]
			X = np.swapaxes(X, 0, 2).reshape(X.shape[2], X.shape[0]*X.shape[1])
		
		tsne = TSNE(perplexity=50, learning_rate=100)
		X_embedded = tsne.fit_transform(X)
		
		x = [i[0] for i in X_embedded]
		y = [i[1] for i in X_embedded]
		
		plt.scatter(x, y, c=colors)

		try:
			os.makedirs(os.path.join(path, 'TSNE'))
		except:
			FileExistsError
		
		i = 0
		while os.path.exists(os.path.join(path, 'TSNE', 'TSNE_trials_{}_{}_{}.png'.format(dataset, name, i))):
			i += 1
		plt.savefig(os.path.join(path, 'TSNE', 'TSNE_trials_{}_{}_{}_{}.png'.format(dataset, day, name, i)))
		plt.close()

def correlation_clustering(acti, meta_df, name, path, be='CR'):
	for day in set(meta_df['Day'].tolist()):
		tmp_meta_df = meta_df[meta_df['Day'] == day]
		for block in set(meta_df['Block'].tolist()):
			tmp_meta_df = meta_df[meta_df['Block'] == block]
			trials_of_interest = tmp_meta_df[tmp_meta_df['Behavior'] == be].index.tolist()
			if len(trials_of_interest) > 0:
				a = list(set(meta_df.loc[meta_df.Block == block, 'Performance'].values.tolist()))[0]
				if a == max(list(set(meta_df.loc[meta_df['Day'] == day, 'Performance'].tolist()))):
					corrs = np.array([np.corrcoef(acti[:, :, trial]) for trial in trials_of_interest])
					corr_matrix = np.mean(corrs, axis=0)
					fig = sns.clustermap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
					indexes = fig.dendrogram_col.reordered_ind

					plt.title('Learning_state : {0}  | Day : {1}'.format(set(meta_df.loc[meta_df.Block == block, 'Performance'].values.tolist()), 
														   				 set(meta_df.loc[meta_df.Block == block, 'Day'].values.tolist())))
					try:
						os.makedirs(os.path.join(path, 'Cluster'))
					except:
						FileExistsError
					
					i = 0
					while os.path.exists(os.path.join(path, 'Cluster', 'correlation_cluster_{}_{}_{}_{:02d}.png'.format(block, be, name, i))):
						i += 1

					fig.savefig(os.path.join(path, 'Cluster', 'correlation_cluster_{}_{}_{}_{:02d}.png'.format(block, be, name, i)))
					
					plt.clf()
					plt.close()

		for block in set(meta_df['Block'].tolist()):
			tmp_meta_df = meta_df[meta_df['Block'] == block]
			trials_of_interest = tmp_meta_df[tmp_meta_df['Behavior'] == be].index.tolist()
			if len(trials_of_interest) > 0:
				a = list(set(meta_df.loc[meta_df.Block == block, 'Performance'].values.tolist()))[0]
				if not a == max(list(set(meta_df.loc[meta_df['Day'] == day, 'Performance'].tolist()))):
					corrs = np.array([np.corrcoef(acti[:, :, trial]) for trial in trials_of_interest])
					corr_matrix = np.mean(corrs, axis=0)
					try:
						corr_matrix = corr_matrix[indexes, :]
						corr_matrix = corr_matrix[:, indexes]

						fig = sns.clustermap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, row_cluster=False, col_cluster=False)
						plt.title('Learning_state : {0}  | Day : {1}'.format(set(meta_df.loc[meta_df.Block == block, 'Performance'].values.tolist()), 
															   				 set(meta_df.loc[meta_df.Block == block, 'Day'].values.tolist())))
						
						fig.savefig(os.path.join(path, 'Cluster', 'correlation_cluster_{}_{}_{}_{:02d}.png'.format(block, be, name, i)))
					except:
						UnboundLocalError
					
					plt.clf()
					plt.close()
