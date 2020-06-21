import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from automata_1 import Cellular_automata
from scipy.ndimage import measurements 
import matplotlib.animation as animation 

data = pd.read_csv("SP500.csv")

closure_price = data[data.columns[1]]
closure_price = closure_price.to_numpy().astype(float)

plt.xlabel("Tiempo [dias]")
plt.ylabel("Precio [UA]")
plt.title("S&P 500")
plt.plot(closure_price, "b")
plt.savefig("imagen1_presentation", transparent = True)



### Evidence the hoshen-kopelman algorithm

size = 64
A = 1.0
a = 0.001
h = 0
steps = 254

fig = plt.figure()

system = Cellular_automata(size,A,a,h)
p_Buy= 0.1; p_Sell = 0.1
system.make_grid(p_Buy,p_Sell)
system.make_clusters()

ims = []
price = []; price.append(system.x())
x_price = np.arange(steps)

for i in range(steps):
	print("progress = \033[91m {:.2f}%\033[0m".format((i/steps)*100)+"\r",end ="")
	system.make_clusters()
	system.update_grid_ising()
	price.append(system.x())
	im, = plt.plot(x_price[:i], price[:i], color='black')
	title_im = plt.text(4*size/10,size,"Cluster = {}".format(system.n_clusters))

	ims.append([im,title_im])

ani = animation.ArtistAnimation(fig,ims, interval = 500, blit = False)
ani.save("hoshen-kopelman.gif", dpi = 80, writer = "imagemagick")
plt.show()
