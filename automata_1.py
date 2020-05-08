import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

class Cellular_automata(object):
	"""Initial definition of the world. Determine the dimensions"""
	def __init__(self, size,Aa,a,hh):
		self.size = size
		self.Aa = Aa
		self.a = a
		self.hh = hh
		

	def make_grid(self, p):
		"""
		Method to construct the grid, p refers to the probability
		to find ones
		"""
		# This procedure generates only a random matrix with ones and zeros, it's wrong !
		# but kept this
		#self.grid = np.random.rand(self.size,self.size)
		#self.grid = np.where(self.grid >= p ,0,1)
		p_Buy, p_Sell = p*np.random.random_sample(2)
		self.grid = np.random.choice([-1,1,0], size = (self.size,self.size),p=[p_Buy,p_Sell,1-p_Sell-p_Buy])


	def show_grid(self):
		selling = np.sum(self.grid > 0)
		buying = np.sum(self.grid < 0)
		inactive = self.size**2-selling - buying

		values = np.unique(self.grid.ravel())
		plt.figure(figsize =(10,8))
		im = plt.imshow(self.grid, interpolation = None)
		colors = [im.cmap(im.norm(value)) for value in values]
		patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
		plt.legend(handles = patches,loc = 4, borderaxespad = 0.)
		plt.xlabel("x")
		plt.xlabel("y")
		plt.title("selling ={}, buying = {}, inactive ={}".format(selling,buying,inactive))
		plt.savefig("percolation")
		plt.show()
		print("state report")

	def state(self):
		"""
		Show the current state of the matriz
		First = active cell
		Second = death cell
		"""
		live = np.sum(self.grid > 0)
		death = self.size**2-np.sum(self.grid>0)
		#print("system state, live = {}, death = {}".format(live,death))
		return live, death

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

		#self.grid = new_label(clust,labels)
		self.grid_label = new_label(clust,labels) 

		# Clusters numbers
		self.n_clusters = len(np.unique(self.grid_label))-1
		# Matrix of index number for a k-cluster
		# first index refers to k-clster
		# second to find coordinate in cluster matrix
		self.index = []
		for k in range(self.n_clusters):
			s = np.where(self.grid_label == np.unique(self.grid_label)[k])
			self.index.append(zip(s[0],s[1]))

		self.index = np.array(self.index)

		# Random variables to define A
		# xi lives in [0,clusters number]
		self.xi = np.random.uniform(-1,1,self.n_clusters)
		# eta lives in [0,spins_numbers in k cluster number :0,k-cluster number]
		self.eta = []
		for k in range(self.n_clusters):
			self.eta.append(np.random.uniform(-1,1,(len(self.index[k]),len(self.index[k]))))
		self.eta = np.array(self.eta)
		#print("Cluster numbers = {}".format(self.n_clusters))
		# zita lives in [0, clusters number]
		self.zita = np.random.uniform(-1,1,self.n_clusters)

	def A(self,k,i,j):
		rA = self.Aa*self.xi[k] + self.a*self.eta[k][i][j]
		return rA

	def h(self,k,i):
		return self.hh*self.zita[k]

	def sigma(self,k,i):
		return self.grid[self.index[k][i]]
	
	def I(self,k,i):
		I_aux = 0
		Nk = len(self.index[k])
		for j in range(Nk):
			if k == 0:
				I_aux += self.A(k,i,j)*(self.sigma(k,j)) + self.h(k,i) 
			else:
				I_aux += self.A(k,i,j)*(self.sigma(k,j)/k) + self.h(k,i) 
		return I_aux/Nk

	def p(self,k,i):
		"""
		The probability is determined by analogy to heat bath dynamics 
		with formal temperature k_b*T = 1
		"""
	
		# Atention with the espin of the 0-cluster and 0-trader !!
		return 1./(1+np.exp(-2*self.I(k,i)))

	def update_traders(self):
		for k in range(self.n_clusters):
			for i in range(len(self.index[k])):
				if self.grid[self.index[k][i]] != 0:
					if self.p(k,i) > 0.5:
						self.grid[self.index[k][i]] = 1
					if  self.p(k,i) == 0.5:
						self.grid[self.index[k][i]] = self.grid[self.index[k][i]] 
					if self.p(k,i) < 0.5:
						self.grid[self.index[k][i]] = -1

			print("update successful")
	

	def update_grid(self):
		pass

	def x(self):
		"""
		Weighted average for the orientation of the spins 
		Ncl : the number of clusters on the grid
		Nk : the size of the kth cluster
		beta: normalization constant
		"""
		X = 0; N = 0
		Ncl = self.n_clusters
		for k in range(Ncl):
			Nk = len(self.index[k])
			for i in range(Nk):
				if k == 0:
					X += Nk*self.sigma(k,i)
					N += self.sigma(k,i)
				else:
					X += Nk*self.sigma(k,i)/k
					N += self.sigma(k,i)/k
		return float(X)/N

	def R(self,t):
		Rt = []
		for i in range(t):
			Po = self.x()
			self.update()
			Pt = Po*(1+self.x())
			Rt.append(np.log(Pt)-np.log(Po))
			print(i+" step")
		Rt = np.array(Rt)
		return (Rt-np.mean(Rt))/np.std(Rt)
		
		











	
"""
	def update_GOL(self,model):
		
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









		
"""