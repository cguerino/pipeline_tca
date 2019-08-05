import os 
import numpy as np 

from functions import analysis as an 
from functions import data
from functions import settings as sett 

params = sett.params()
paths = sett.paths()
ar = sett.arguments()

args = ar.get_arguments()
selection = ar.get_selection()
tca_sett = ar.get_tca_sett()
animal = ar.get_animal()

name = ''.join(['_' + k + '-' + str(selection[k]) for k in selection if not selection[k] == None])[1:]
path = os.path.join(paths.path2Output, animal, args.function, args.init, str(args.rank), name)
path_fig = os.path.join(paths.path2Figures, animal, args.function, args.init, str(args.rank), name)
for p in [path, path_fig]:
	try: 
		os.makedirs(p)
	except:
		FileExistsError


meta_df, roi_tensor, acti, norm_acti, smoothed_acti = data.load_processed_data_all(animal)
meta_df, acti, norm_acti, smoothed_acti = data.select_data(meta_df, acti, norm_acti, smoothed_acti, selection)

if args.similarity:
	sim = an.similarity_kruskal(fac1, fac2)
	if args.verbose: print(sim)

if args.TSNE:
	an.TSNE_trials(acti, meta_df, name, path_fig, dataset=args.TSNE)

if args.correlation:
	for b in ['CR', 'HIT', 'MISS', 'FA']:
		an.correlation_clustering(acti, meta_df, name, path_fig, be=b)


###¨Put predict 