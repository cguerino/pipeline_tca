import os

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
