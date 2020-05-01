import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 

# Parameters 
size = 128
A = 1.0
a = 0.001
h = 0

# Class constructor
aut = Cellular_automata(size,A,a,h)
#initial state
aut.make_grid(p = 0.5)
aut.make_clusters()


plt.imshow(aut.grid, cmap = "nipy_spectral")
plt.colorbar()
plt.savefig("percolation")
plt.xlabel("x")
plt.xlabel("y")
plt.title("Grid Traders")
plt.show()
