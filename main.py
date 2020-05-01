import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np 

# Parameters 
size = 4
A = 1.0
a = 0.001
h = 0
P = np.zeros((size,size))

# Class constructor
aut = Cellular_automata(size,A,a,h)
#initial state
aut.make_grid(p = 0.3)
aut.make_clusters()

# Construct the probability matrix

