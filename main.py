import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 
from utils import plot_all, logarithmic_price
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from scipy.ndimage import measurements
import pandas as pd


# ---------- Read the index data ---------- #

plot_data = False

if plot_data == True:
	#data = np.loadtxt("index.csv")
	data = pd.read_csv("SP500.csv")

	closure_price = data[data.columns[1]]
	closure_price = closure_price.to_numpy().astype(float)
	
	fecha = data[data.columns[0]]
	fecha = fecha.to_numpy()

	price = logarithmic_price(closure_price)

	plt.plot(price)
	plt.show()

	# Plot and save all data
	# plot_all(data)



# ---------- Evidence the algorithm ---------- #

execute = True

if execute == True:
	

	# ---------- Review    of   update ---------- #

	#system parameters
	size = 128
	A = 1.0
	a = 0.001
	h = 0
	steps = 10

	Automata1 = Cellular_automata(size,A,a,h)
	p_Buy= 0.1; p_Sell = 0.1
	Automata1.make_grid(p_Buy,p_Sell)
	Automata1.make_clusters()
	fig = plt.figure(figsize = (15,5))
	ax1 = fig.add_subplot(1,2,1)
	#ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)
	ax2.set_xlabel("Jornada de Negociacion [t]")
	ax2.set_ylabel("Precio [UA]")
	ax2.set_ylabel("Precio [UA]")
	#ax2.set_xlim(10,steps)


	ims = []
	price = []; price.append(Automata1.x())
	x_price = np.arange(steps)	

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
		Automata1.Random_variables()
		Automata1.update_traders()
		price.append(Automata1.x())
		line2,  = ax2.plot(x_price[:i], price[:i], color='black')
		
	
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
		
		ims.append([im,title_im,title_im2,traders,line2])
		ims[0][3:]
		n +=1
		
	ani = animation.ArtistAnimation(fig,ims,interval = 500,blit = False)
	#ani.save("animacion2.gif",dpi=80,writer="imagemagick")
	plt.show()


# ---------- Review the price index of simulation ---------- #

# size = 8
# A = 1.0
# a = 0.001
# h = 0
# steps = 500

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)

# #ax1.set_ylim(0, 5000)
# #ax2.set_ylim(0, 5000)



# system_price = Cellular_automata(size,A,a,h)
# p_Buy= 0.3; p_Sell = 0.25
# system_price.make_grid(p_Buy,p_Sell)
# system_price.make_clusters()

# lines = []
# price = []
# x_price = np.arange(steps)


# for i in range(steps):
# 	print("progress = \033[91m {:.2f}%\033[0m".format((i/steps)*100)+"\r",end ="")
# 	system_price.update_grid_ising()
# 	#system_price.update_traders()
# 	price.append(system_price.x())
# 	line1,  = ax1.plot(x_price[:i], price[:i], color='black')
# 	line2,  = ax2.plot(x_price[:i], price[:i], color='red')
# 	lines.append([line1,line2])

# ani = animation.ArtistAnimation(fig,lines,interval=50,blit=True)
# ani.save("precio.gif",dpi=80,writer="imagemagick")
# plt.show()








