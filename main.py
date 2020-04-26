import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 
# Parameters 
size = 128

# Class constructor
aut = Cellular_automata(size)
#initial state

aut.state()
aut.make_cluster(size/2)
aut.make_cluster(size/3)
#plt.imshow(aut.grid, cmap = plt.cm.Blues)
#plt.show()

#for i in range(20):
#	aut.update()
#	aut.state()

#	plt.imshow(aut.grid, cmap = plt.cm.Blues)
	#plt.savefig("im_"+str(1))
#	plt.show() 

print(np.random.uniform(-1,1,1))

