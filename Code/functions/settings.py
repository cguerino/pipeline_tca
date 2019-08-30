import os
import argparse

class paths():
	""" Class containing all paths. We should verify if these are correct each time you change system

	Attributes
	----------
	path2Data : str
		Path to Data folder
	path2Output : str
		Path to Output folder
	path2Figures : str
		Path to Figures folder
	path2fixed_factors : str
		Path to fixed_factors folder. Use only when doing fixed parafac.

	Constructors
	------------
	__init__(self)
	
	"""
	def __init__(self):
		""" Constructor at initialization
		"""
		self.path2Data = os.path.join(os.sep, 'C:' + os.sep, 'Users', 'Antonin', 'Documents',  'Data')
		self.path2Output = os.path.join(os.sep, 'C:' + os.sep, 'Users', 'Antonin', 'Documents',  'Output')
		self.path2Figures = os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Figures')
		self.path2fixed_factors = os.path.join(os.sep, 'X:' + os.sep, 'Antonin', 'Pipeline', 'Data', 'fixed_factors')

class params():
	""" Class containing all parameters for overall experiment
	
	Attributes
	----------
	animal_list : list
		List containing all animal names. These names must be identical to the folder name of the data. 
		To add a animal, just add its name at the end of the list

	Constructors
	------------
	__init__(self)

	"""
	def __init__(self):
		"""Constructor at initialization
		"""
		self.animal_list = ['012216-02', '012216-03', '012216-04', '012216-06',
			               '082216-01', '082216-02', '082216-03', '082216-05',
			               '020817-03', '020817-04','020618-01', 
			               '020618-04', '020618-06', '020618-09', '020618-10'] #071114-06

class arguments():
	"""Class containing all arguments that can be passed to the Pipeline. As each file call this class,
	You are able at any time to pass/add arguments from here

	Attributes
	----------
	args : object
		Object encapsulating all arguments
	selection : dict
		Dictionnary containng arguments related to selection criteria
	tca_sett : dict 
		Dictionnary containng arguments related to TCA parameters 
	preprocess_sett : dict
		Dictionnary containing arguments related to preprocessing parameters
		Animal's name
	fixed_selection : dict
		Dictionnary containing arguments related to fixed time factor generation

	Constructors
	------------
	__init__(self)

	Methods
	-------
	get_arguments()
		Get all arguments
	get_selection()
		Get arguments related to selection criteria of data
	get_tca_sett()
		Get arguments related to TCA
	get_preprocess_sett()
		Get arguments related to preprocessing
	get_animal()
		Get animal's name from index in the list
	get_fixed_args()
		Get arguments related to selection criteria for fixed parafc time factor generation

	"""
	def __init__(self):
		""" Constructor at initialization
		"""
		parser = argparse.ArgumentParser(description='Parameters for computing')

		parser.add_argument('--plotting', '-p', action='store_true', 
							help='Plot figures during processing')
		parser.add_argument('--verbose', '-v', action='store_true', 
							help='Display detailed output in prompt')
		parser.add_argument('--cutoff', '-co', type=int, default=25, 
							help='Cut-off for consecutive NaNs in a trial')
		parser.add_argument('--thres', '-th', type=int, default=80, 
							help='Threshold for non_consecutive NaNs in a trial')
		parser.add_argument('--animal', '-a', nargs='*', type=int, default=15, 
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
							choices=['non_negative_parafac', 'parafac', 'custom_parafac', 'non_negative_fixed_parafac', 'fixed_prafac'], 
							help='TCA function to use: parafac, non_negative_parafac, custom_parafac')
		parser.add_argument('--neg_fac', '-nf', type=str, default=0, choices=[0, 1, 2],
							help='Factor of the TCA which is allowed to be negative')
		parser.add_argument('--init', '-i', type=str, default='random', choices=['random', 'svd'],
							help='Initialization method for parafac')
		parser.add_argument('--norm', '-n', type=str, default='min_max', choices=['norm', 'min_max', 'day'],
							help='Normalization_method')
		parser.add_argument('--similarity', '-sim', action='store_true', 
							help='Compute similarity score between two kruskal matrices')
		parser.add_argument('--TSNE', '-ts', type=str, default=None, 
							help='Generate TSNE figure')
		parser.add_argument('--correlation', '-corr', action='store_true', 
							help='Generate hierarchical correlation clstering figure between two trials')
		parser.add_argument('--fixed_factor', '-ff', type=str, default=None, 
							help='Name of fixed_prafac npy file in fixed_factors folder to use for fixed parafac')
		
		# Select relevant data for processing. See meta_df files to see valid values for each animal
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

		# Select relevant data for fixed parafac
		parser.add_argument('--fExpclass', '-fE', type=int, nargs='*', default=None, 
							help='Experiment classes to choose from. All experiment classes are taken by default')
		parser.add_argument('--fanimal', '-fa', type=int, nargs='*', default=None, 
							help='Animal number for fixed parafac')
		parser.add_argument('--name', '-fn', type=str, default=None, 
							help='Name of fixed_prafac npy file')
		self.args = parser.parse_args()

	def get_arguments(self):
		""" Get all arguments
		
		Returns
		-------
		dict
			All arguments
		"""
		return self.args

	def get_selection(self):
		""" Get arguments related to selection criteria
		
		Returns
		-------
		dict
			Arguments related to selection criteria
		"""
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
		""" Get arguments related to TCA settings
		
		Returns
		-------
		dict
			Arguments related to TCA settings
		"""
		self.tca_sett = {
		'Animal': self.args.animal,
		'Rank': self.args.rank,
		'Function': self.args.function,
		'Neg_Fac': self.args.neg_fac, 
		'Init': self.args.init,
		}
		return self.tca_sett

	def get_preprocess_sett(self):
		""" Get arguments related to preprocessing settings

		Returns
		-------
		dict
			Arguments related to preprocessing settings
		"""
		self.preprocess_sett = {
		'Cut_off': self.args.cutoff,
		'Threshold': self.args.thres,
		'Animal': self.args.animal,
		'Tol': self.args.tol,
		'Norm': self.args.norm
		}

		return self.preprocess_sett

	def get_animal(self):
		""" Get animal name from index

		Returns
		-------
		str
			Animal's name
		"""
		self.animal = params().animal_list[self.args.animal[0]]

		return self.animal

	def get_fixed_args(self):
		""" Get arguments related to fixed parafac settings

		Returns
		-------
		dict
			Arguments related to fixed parafac
		"""
		self.fixed_selection = {
		'Animal': self.args.fanimal,
		'Experiment Class': self.args.fExpclass
		}

		return self.fixed_selection


