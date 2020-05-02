import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 

# Parameters 
size = 16
A = 1.0
a = 0.001
h = 0
P = np.zeros((size,size))

# Class constructor
aut = Cellular_automata(size,A,a,h)
#initial state
aut.make_grid(p = 0.5)
aut.make_clusters()
print(aut.grid)
print(aut.update(10))
print(aut.grid)





