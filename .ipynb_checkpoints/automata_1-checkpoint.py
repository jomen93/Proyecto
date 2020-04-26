import numpy as np

class Cellular_automata():
	"""docstring for Cellular_automata"""
	def __init__(self, size_x, size_y, n_agents, N_clusters):
		self.size_x = size_x
		self.size_y = size_y
		self.n_agents = n_agents
		self.N_clusters = N_clusters
	

	def Grid(self):
		self.grid = np.zeros((self.size_x,self.size_y))
		return self.grid




	      	
	#def make_cluster(self,self.s):
		