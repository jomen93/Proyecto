import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 
from utils import plot_all
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from scipy.ndimage import measurements


# size = 16
# A = 1.0
# a = 0.001
# h = 0
# system = Cellular_automata(size,A,a,h)
# p_Buy= 0.4; p_Sell = 0.4
# system.make_grid(p_Buy,p_Sell)
# system.make_clusters()


# print(system.grid_label)
# index = []
# print(system.n_clusters)

# #for k in range(system.n_clusters):
# k = 0
# i = 0

# def index(k,i):
# 	s = list(np.argwhere(system.grid_label == k))[i]
# 	return s[0],s[1]

# print(index(0,0))
# print(system.grid[index(0,0)])

# ---------- Read the ndex data ---------- #
"""
Review of the Hoshen-Kopelman algorithm
"""



Review = False
if Review == True:
	size = 16
	A = 1.0
	a = 0.001
	h = 0
	steps = 10
	system = Cellular_automata(size,A,a,h)
	p_Buy= 0.4; p_Sell = 0.4
	system.make_grid(p_Buy,p_Sell)
	plt.imshow(system.grid,origin='lower',interpolation='nearest',cmap= "RdGy")
	plt.colorbar()
	plt.show()
	system.make_clusters()

	lw, num = measurements.label(system.grid_label)
	b = np.arange(lw.max() + 1)
	np.random.shuffle(b)
	shuffledLw = b[lw]
	plt.imshow(shuffledLw,origin='lower',interpolation='nearest',cmap = "RdGy")
	plt.colorbar()
	plt.show()
	print(system.n_clusters)

	system.update_grid_ising()

	lw, num = measurements.label(system.grid_label)
	b = np.arange(lw.max() + 1)
	np.random.shuffle(b)
	shuffledLw = b[lw]
	plt.imshow(shuffledLw,origin='lower',interpolation='nearest',cmap = "RdGy")
	plt.colorbar()
	plt.show()
	print(system.n_clusters)
	

# ---------- Read the ndex data ---------- #
plot_data = True

if plot_data == True:
	data = np.loadtxt("index.csv")

	# Plot and save all data
	plot_all(data)



# ---------- Evidence the algorithm ---------- #

execute = True

if execute == True:
	

	# ---------- Review    of   update ---------- #

	#system parameters
	size = 64
	A = 1.0
	a = 0.001
	h = 0
	steps = 100

	Automata1 = Cellular_automata(size,A,a,h)
	p_Buy= 0.3; p_Sell = 0.25
	Automata1.make_grid(p_Buy,p_Sell)
	Automata1.make_clusters()
	fig = plt.figure(figsize=(8,5))
	ax1 = fig.add_subplot(1,1,1)
	#ax1 = fig.add_subplot(1,2,1)
	#ax2 = fig.add_subplot(1,2,2)

	ims = []

	n = 0
	for i in range(steps):
		print("progress = \033[91m {:.2f}%\033[0m".format((i/steps)*100)+"\r",end ="")
		Automata1.make_clusters()
		lw, num = measurements.label(Automata1.grid_label)
		b = np.arange(lw.max() + 1)
		np.random.seed(42)
		np.random.shuffle(b)
		shuffledLw = b[lw]
		
		Automata1.update_grid_ising()
		#Automata1.Random_variables()
		Automata1.update_traders()
		
	
		values = np.unique(Automata1.grid.ravel())

		im = ax1.imshow(Automata1.grid,origin='lower',interpolation='nearest',cmap= "viridis",animated = True)
		title_im = ax1.text(0.2*size,size,"Paso {}".format(i))
		#im2 = ax2.imshow(shuffledLw,origin='lower',interpolation='nearest',cmap= "binary",animated = True)
		# im2 = ax2.imshow(Automata1.grid_label,origin='lower',interpolation='nearest',cmap= "binary",animated = True)
		title_im2 = ax1.text(0.7*size,size,"Clusters ={}".format(Automata1.n_clusters))
		
		# show traders
		selling = np.sum(Automata1.grid > 0)
		buying = np.sum(Automata1.grid < 0)
		inactive = Automata1.size**2-selling - buying

		traders = ax1.text(0.05*size,size+0.05*size,"vendedores {}   compradores {}   inactivos {}".format(selling,buying,inactive))


		#colors = [im.cmap(im.norm(value)) for value in values]
		#patches = [ mpatches.Patch(color=colors[i], label="Estado {l}".format(l=values[i]) ) for i in range(len(values)) ]
		#plt.legend(handles = patches,loc = 4, borderaxespad = 0.)
		
		#multi animation
		#ims.append([im,title_im,im2,title_im2,traders])
		ims.append([im,title_im,title_im2,traders])
		n +=1
		
	ani = animation.ArtistAnimation(fig,ims,interval = 500,blit = False)
	ani.save("animacion.gif",dpi=80,writer="imagemagick")
	plt.show()



