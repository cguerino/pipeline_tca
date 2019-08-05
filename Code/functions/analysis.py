import os
import numpy as np
import pandas as pd 
from functions import data

from functions import settings as sett 
from functions import tca_utils as tca
from functions import training as train

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm 

from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

paths = sett.paths()


def similarity_kruskal(fac1, fac2):
	sim = tca.kruskal_align(fac1, fac2, permute_U=True, permute_V=True)

	return sim

def TSNE_trials(acti, meta_df, name, path, dataset='factors'):
	days = list(set(meta_df['Day'].tolist()))
	blocks = list(set(meta_df['Block'].tolist()))

	for day in days:
		trials_of_interest = meta_df[meta_df.Day == day].index.tolist()
		colors = meta_df.loc[trials_of_interest, 'Performance Color'].tolist()
		
		if dataset == 'factors':
			X = factors[2][trials_of_interest, :]
		elif dataset == 'acti':
			X = acti[:, :, trials_of_interest]
			X = np.swapaxes(X, 0, 2).reshape(X.shape[2], X.shape[0]*X.shape[1])
		
		tsne = TSNE(perplexity=10, learning_rate=200)
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
