import numpy as np
import matplotlib.pyplot as plt 

class Cellular_automata(object):
	"""Initial definition of the world. Determine the dimensions"""
	def __init__(self, size,Aa,a,hh):
		self.size = size
		self.Aa = Aa
		self.a = a
		self.hh = hh


	def make_grid(self, p = 0.5):
		"""
		Method to construct the grid, p refers to the probability
		to find ones
		"""
		self.grid = np.random.rand(self.size,self.size)
		self.grid = np.where(self.grid >= p ,0,1)

	def show_grid(self):
		live = np.sum(self.grid > 0)
		death = self.size**2-np.sum(self.grid>0)

		plt.imshow(self.grid, cmap = "nipy_spectral")
		plt.colorbar()
		plt.savefig("percolation")
		plt.xlabel("x")
		plt.xlabel("y")
		plt.title("Grid Traders live ={} death = {}".format(live,death))
		plt.show()


	def state(self):
		"""
		Show the current state of the matriz
		First = active cell
		Second = death cell
		"""
		live = np.sum(self.grid > 0)
		death = self.size**2-np.sum(self.grid>0)
		print("system state, live = {}, death = {}".format(live,death))
		pass

	def make_clusters(self):
		"""
		Implementation of the Hoshen-Kopelman algorithm:
		Firts span the lattice once. Each time we find occupied cell, we
		check the neighbors at the top and left of the current cell. We 
		have then four pssibilities:

		1. Both cell are empty: we create a new cluster label and 
		remplace to the current cell 
		2. Only one cell is ocupied: we set the cluster label of the
		occupied cell to the current cell
		3. Both cells are ocupied and have same cluster label: we set 
		this cluster label to the current cell
		4. If Both cells are occupied but have distintc cluster labels, 
		we set the smallest to be the current cell cluster label and we 
		add the union between both cluster labels as a new wnrey into a
		labels

		"""
		n_clust = 0
		labels = np.arange(self.size**2)
		clust = np.zeros((self.size,self.size),dtype = int)

		def find(m, labels):
			y = int(m)
			while labels[y] != y:
				y = labels[y]
			while labels[int(m)] != int(m):
				z = labels[int(m)]
				labels[int(m)] = y
				m = z
			return y, labels
		
		def union(m,n,labels):
			labels[int(find(m,labels)[0])] = int(find(n,labels)[0])
			return labels

		def new_label(clust,labels):
			nlabels = np.zeros(self.size**2)
			for i in range(0,self.size):
				for j in range(0,self.size):
					if clust[i,j] != 0:
						x, labels = find(clust[i,j],labels)
						if nlabels[x] == 0:
							nlabels[0] += 1
							nlabels[x] = nlabels[0]
						clust[i,j] =nlabels[x]
			return clust

		for i in range(0,self.size):
			for j in range(0,self.size):
				if self.grid[i,j] == 1:
					if i == 0 and j == 0:
						n_clust += 1
						clust[i,j] = n_clust
					elif i == 0 and j!= 0:
						if self.grid[i,j-1] == 0:
							n_clust +1
							clust[i,j] = n_clust
						else:
							clust[i,j], labels = find(clust[i,j-1],labels)
					elif i != 0 and j == 0:
						if self.grid[i-1,j] == 0:
							n_clust += 1
							clust[i,j] = n_clust
						else:
							clust[i,j], labels = find(clust[i-1,j],labels)
					else:
						if self.grid[i-1,j] == 0 and self.grid[i,j-1] == 0:
							n_clust +=1
							clust[i,j] = n_clust
						elif self.grid[i-1,j] != 0 and self.grid[i,j-1] == 0:
							clust[i,j],labels = find(clust[i-1,j],labels)
						elif self.grid[i-1,j] == 0 and self.grid[i,j-1] != 0:
							clust[i,j],labels = find(clust[i,j-1],labels)
						else:
							labels = union(clust[i,j-1],clust[i-1,j],labels)
							clust[i,j],labels = find(clust[i-1,j],labels)

		self.grid = new_label(clust,labels)
		self.n_clusters = len(np.unique(self.grid))-1
		self.xi = np.random.uniform(-1,1,self.n_clusters)

		print("Cluster numbers = {}".format(self.n_clusters))

	def index(self,k):
		s = np.where(self.grid == np.unique(self.grid)[k])
		s = zip(s[0],s[1])
		return s

	def A(self,k,i,j):
		eta = np.random.uniform(-1,1,(self.n_clusters,self.n_clusters))
		return self.Aa*self.xi[k] + self.a*eta[i,j]

	def h(self,k,i):
		zita = np.random.uniform(-1,1,(self.n_clusters))
		return self.hh*zita[k]

	def sigma(self,k,i):
		return self.grid[self.index(k)[i]]
	
	def I(self,k,i):
		I_aux = 0
		Nk = len(self.index(k))
		for j in range(1):
			I_aux += self.A(k,i,j)*self.sigma(k,j) + self.h(k,i) 
		return I_aux/Nk

	def p(self,k,i):
		"""
		The probability is determined by analogy to heat bath dynamics 
		with formal temperature k_b*T = 1
		"""
		#self.node_index = np.where(self.grid == np.unique(self.grid)[k])
		#self.node_index = zip(self.node_index[0],self.node_index[1])

		# Atention with the espin of the 0-cluster and 0-trader !!
		return 1./(1+np.exp(-2*self.I(k,i)))

	def update(self,k):
		
		for i in range(len(self.index(k))):
			print(self.p(k,i))
			if self.p(k,i) >= 0.5:
				self.grid[self.index(k)[i]] = k
			else:
				self.grid[self.index(k)[i]] = 0
		


# Estoy pasando la funcion update


	def x(self):
		pass
		





	def update1(self,model):
		"""
		** Game of life **
		Examine the number of neighbors for each cell of the matriz 
		and determine the future of the cell. 1 or 0 

		Algorithm 

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


		 		if model == "Game of Life":
			 		# Automata Rules of Game of life

			 		if self.grid[i,j] == 1:
		 				if s == 3 or s == 2:
							self.Ngrid[i,j] = 1
						else:
							self.Ngrid[i,j] = 0

					if self.grid[i,j] == 0:
						if s == 3:
				 			self.Ngrid[i,j] = 1
				 		else:
				 			self.Ngrid[i,j] = 0

		self.grid = self.Ngrid









		
