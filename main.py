import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 
from utils import plot_all
import matplotlib.animation as animation



# ---------- Read the ndex data ---------- #
plot_data = False

if plot_data == True:
	data = np.loadtxt("index.csv")

	# Plot and save all data
	plot_all(data)



# ---------- Evidence the Cluster tagging algorithm ---------- #

execute = True

if execute == True:
	

	# ---------- Review of traders update ---------- #

	size = 64
	A = 1.0
	a = 0.001
	h = 0

	Automata1 = Cellular_automata(size,A,a,h)
	p_Buy= 0.3; p_Sell = 0.2
	Automata1.make_grid(p_Buy,p_Sell)
	Automata1.make_clusters()
	ims = []
	fig = plt.figure()
	for i in range(50):
		print("progress = "+str((i/50.)*100)+"%\r",end ="")
		#Automata1.show_grid("three_"+str(i))
		Automata1.update_grid_ising()
		#Automata1.update_traders()
		#Automata1.market_dynamics()
		im = plt.imshow(Automata1.grid,animated = True)
		ims.append([im])
		
	ani = animation.ArtistAnimation(fig,ims,interval = 200)
	ani.save("animacion.gif",dpi=80,writer="imagemagick")
	plt.show()



