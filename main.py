#import matplotlib 
#matplotlib.use("TKAgg")
import matplotlib.pyplot as plt 
#import matplotlib.animation as animation 
from automata_1 import Cellular_automata
import numpy as np 

from utils import plot_all

# ---------- Read the ndex data ---------- #
data = np.loadtxt("index.csv")

# Plot and save all data
plot_all(data)




# ---------- Evidence the Cluster tagging algorithm ---------- #

execute = False

if execute == True:
	# Parameters 
	size = 128
	A = 1.0
	a = 0.001
	h = 0

	# Object constructor
	Automata = Cellular_automata(size,A,a,h)

	# Make random matrix with 0,1,-1
	# And probability p to fin traders
	p = 0.8
	Automata.make_grid(p)

	# Cluster tagging
	Automata.make_clusters()
	# Show the system
	fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,8))

	selling = np.sum(Automata.grid > 0)
	buying = np.sum(Automata.grid < 0)
	ax1.imshow(Automata.grid)
	ax1.set_title("selling ={}, buying = {}".format(selling,buying))

	ax2.imshow(Automata.grid_label)
	ax2.set_title("Clusters number = {} ".format(Automata.n_clusters))
	plt.show()



	 # ---------- Review of traders update ---------- #

	size = 16
	A = 1.0
	a = 0.001
	h = 0

	Automata1 = Cellular_automata(size,A,a,h)
	p = 0.3
	Automata1.make_grid(p)
	Automata1.make_clusters()

	fig = plt.figure()

	for i in range(60):
		Automata1.update_traders()
		plt.imshow(Automata1.grid)
		plt.savefig("update_traders_"+str(i))






