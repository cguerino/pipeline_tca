import os

for animal in range(15):
	for e in [1, 2, 3]:
		for f in ['non_negative_parafac', 'parafac', 'custom_parafac']:
			os.system('X:\\Antonin\\Pipeline\\Code\\TCA.py -a {} -E {} -f {}'.format(animal, e, f))
			os.system('X:\\Antonin\\Pipeline\\Code\\meta_analysis.py -a {} -E {} -f {} -ts "acti" -corr'.format(animal, e, f))