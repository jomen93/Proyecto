import numpy as np

class Cellular_automata(object):
	"""Initial definition of the world. Determine the dimensions"""
	def __init__(self, size,Aa,a,hh):
		self.size = size
		self.Aa = Aa
		self.a = a
		self.hh = hh


		self.grid = np.random.choice([0, 1], size=(self.size,self.size), p=[.8,.2])
		#self.grid = np.zeros((self.size,self.size))


	def state(self):
		"""
		Show the current state of the matriz
		First = active cell
		Second = death cell
		"""
		live = np.sum(self.grid)
		death = self.size**2-np.sum(self.grid)
		print("system state, live = {}, death = {}".format(live,death))
		pass

	def make_cluster(self,center):
		square_mas = int(center+center/6.)
		square_men = int(center-center/6.)
		self.grid[square_men:square_mas,square_men:square_mas] = np.random.choice([0, 1], 
			size=(square_mas-square_men,square_mas-square_men), p=[.2,.8])


	def A(self, i,j):
		xi = np.random.uniform(-1,1,1)
		eta = np.random.uniform(-1,1,(self.size,self.size))
		return self.Aa*xi + self.a*eta[i,j]

	def h(self,i):
		alpha = np.random.uniform(-1,1,1)
		return self.hh*alpha

	def I(self,i):
		I_aux = 0
		for ii in range(self.size):
			for jj in range(self.size):
				I_aux += self.A(i,jj)*self.grid[ii,jj] 
		return (1/self.size**2)*I_aux + self.h(i)

	def p(self,i):
		return (1)/(1+np.exp(-2*self.I(i)))


	def update(self):
		"""
		Examine the number of neighbors for each cell of the matriz 
		and determine the future of the cell. 1 or 0 

		Algprithm 

		1. Choose a cell
		2. Count neighbors
		3. If the cell has 3 neighbors, it lives
		   If the cell is alive and has 2 neighbors , it alives
		   Otherwise, the cell is dead 
		4. replace the grid matriz in the variable Ngrid 
		"""

		self.Ngrid = np.zeros((self.size,self.size))

		#if self.size[0,0]:

		for i in range(self.size):
		 	for j in range(self.size):
		 		# variable to count neighbor each step
		 		s = 0

		 		# Neighbors condition 
		 		for k in [-1,1]:
		 			kx1 = i+k
		 			kx2 = i

		 			ky1 = j
		 			ky2 = j+k
		 			ky3 = j-k

		 			if kx1 < 0: kx1 == self.size - 1
		 			elif kx1 == self.size: kx1 = 0 

					if kx2 < 0: kx2 == self.size - 1
		 			elif kx2 == self.size: kx2 = 0 

		 			if ky1 < 0: ky1 == self.size - 1
		 			elif ky1 == self.size: ky1 = 0 

					if ky2 < 0: ky2 == self.size - 1
		 			elif ky2 == self.size: ky2 = 0 

		 			if ky3 < 0: ky3 == self.size - 1
		 			elif ky3 == self.size: ky3 = 0 		 			

		 			s += self.grid[kx1,ky1]+self.grid[kx2,ky2]
		 			+self.grid[kx1,ky2]+self.grid[kx1,ky3]

		 		# Automata Rules 

		 		if self.grid[i,j] == 1:
	 				if s == 2 or s == 3 or s == 1:
						self.Ngrid[i,j] = 1
					else:
						self.Ngrid[i,j] = 0

				if self.grid[i,j] == 0:
					if s == 3:
			 			self.Ngrid[i,j] = 1
			 		else:
			 			self.Ngrid[i,j] = 0

		self.grid = self.Ngrid









		
