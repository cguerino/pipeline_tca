import os
experiment = 3
function = 'non_negative_parafac'
init='random'
animal=15
odor = ['HEXANONE', 'ETHYL TIGLATE']

for odo in odor:
	os.system('X:\\Antonin\\Pipeline\\Code\\main.py -a {} -E {} -f {} -i {} -v -O "{}" -nf 1'.format(animal, experiment, function, init, odo))