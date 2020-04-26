import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata

# Parameters 
size = 32
A = 1
a = 1
h = 1

# Class constructor
aut = Cellular_automata(size,A,a,h)
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

print(aut.p(1),1-aut.p(1))
