import os
import argparse

class paths():
	def __init__(self):
		self.path2Data = os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline',  'Data')
		self.path2Output = os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Output')
		self.path2Figures = os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Figures')

class params():
	def __init__(self):
		self.animal_list = ['012216-02', '012216-03', '012216-04', '012216-06',
			               '082216-01', '082216-02', '082216-03', '082216-05',
			               '020817-03', '020817-04', '071114-06','020618-01', 
			               '020618-04', '020618-06', '020618-09', '020618-10']
		
		self.color_list =  ['orange', 'green', 'red', 'black', 'blue',
				            'yellow',  'pink', 'cyan', 'purple', 'grey',
				            'grey', 'grey', 'orange', 'green', 'red',
				            'black', 'blue', 'yellow', 'pink']

		self.convert_beh = {'MISS': 0, 'HIT':1, 'FA': 2, 'CR':3}
		self.convert_odo = {'40EB 60AA':0, '60EB 40AA': 1, 'Air2': 2, 'CINEOLE': 3, 
							'ETHYL TIGLATE': 4, 'Min-Limonen': 5, 'VALDEHYDE': 6, 
							'Air1': 7, 'HEXANONE': 8, '55EB 45AA': 9, 'BUTYRIC ACID': 10, 
							'45EB 55AA': 11}

class arguments():

	def __init__(self):
		parser = argparse.ArgumentParser(description='Parameters for computing')

		parser.add_argument('--plotting', '-p', action='store_true', 
							help='Plot figures during processing')
		parser.add_argument('--verbose', '-v', action='store_true', 
							help='Display detailed output in prompt')
		parser.add_argument('--cutoff', '-co', type=int, default=25, 
							help='Cut-off for consecutive NaNs in a trial')
		parser.add_argument('--thres', '-th', type=int, default=80, 
							help='Threshold for non_consecutive NaNs in a trial')
		parser.add_argument('--animal', '-a', type=int, default=15, 
							help='Choose animal to process')
		parser.add_argument('--tol', '-tl', type=int, default=15, 
							help='Tolerance for data')
		parser.add_argument('--preprocess', '-pre', action='store_true', 
							help='Perform preprocessing')
		parser.add_argument('--computation', '-com', action='store_true', 
							help='Perform computation')
		parser.add_argument('--rank', '-r', type=int, default=6, 
							help='Rank of the TCA')
		parser.add_argument('--function', '-f', type=str, default='non_negative_parafac', 
							help='TCA function to use: parafac, non_negative_parafac, custom_parafac')
		parser.add_argument('--neg_fac', '-nf', type=str, default=0, 
							help='Factor of the TCA which is allowed to be negative')
		parser.add_argument('--tmp', '-tmp', action='store_true', 
							help='Save and compute data of tmp folder for debugging')
		parser.add_argument('--init', '-i', type=str, default='random', 
							help='Initialization method for parafac')
		parser.add_argument('--norm', '-n', type=str, default='min_max', 
							help='Normalization_method')
		parser.add_argument('--smooth', '-sm', action='store_true', 
							help='Smooth the data after normalization')
		parser.add_argument('--similarity', '-sim', action='store_true', 
							help='Compute similarity score between two kruskal matrices')
		parser.add_argument('--TSNE', '-ts', type=str, default=None, 
							help='Generate TSNE figure')
		parser.add_argument('--correlation', '-corr', type=str, default=None, 
							help='Generate hierarchical correlation clstering figure between two trials')
		# Select relevant data for processing
		parser.add_argument('--Block', '-B', type=int, nargs='*', default=None, 
							help='Blocks to choose from. All blocks are taken by default')
		parser.add_argument('--Odor', '-O', type=str, nargs='*', default=None, 
							help='Odors to choose from. All odors are taken by default')
		parser.add_argument('--Stimulus', '-S', type=str, nargs='*', default=None, 
							help='Stimulus to choose from. All stimulus are taken by default')
		parser.add_argument('--Behavior', '-BE', type=str, nargs='*', default=None, 
							help='Behaviors to choose from. All behaviors are taken by default')
		parser.add_argument('--Reward', '-R', type=int, default=None, 
							help='Rewards to choose from. All rewards are taken by default')
		parser.add_argument('--Date', '-D', type=str, nargs='*', default=None, 
							help='Dates to choose from. All dates are taken by default')
		parser.add_argument('--Day', '-DY', type=int, nargs='*', default=None, 
							help='Days to choose from. All days are taken by default')
		parser.add_argument('--Learnstate', '-L', type=int, default=None, 
							help='Learning states to choose from. All learning states are taken by default')
		parser.add_argument('--Expclass', '-E', type=int, nargs='*', default=None, 
							help='Experiment classes to choose from. All experiment classes are taken by default')
		parser.add_argument('--Performance', '-P', type=int, nargs='*', default=None, 
							help='Performances to choose from. All performances are taken by default')

		self.args = parser.parse_args()

	def get_arguments(self):
		return self.args

	def get_selection(self):
		self.selection = {
		'Block':self.args.Block, 
		'Odor': self.args.Odor,
		'Stimulus': self.args.Stimulus,
		'Behavior': self.args.Behavior,
		'Reward': self.args.Reward,
		'Date': self.args.Date,
		'Day': self.args.Day,
		'Learning State': self.args.Learnstate,
		'Experiment Class': self.args.Expclass,
		'Performance': self.args.Performance
		}

		return self.selection

	def get_tca_sett(self):
		self.tca_sett = {
		'Animal': self.args.animal,
		'Rank': self.args.rank,
		'Function': self.args.function,
		'Neg_Fac': self.args.neg_fac, 
		'Init': self.args.init,
		}
		return self.tca_sett

	def get_animal(self):
		self.animal = params().animal_list[self.args.animal]

		return self.animal

