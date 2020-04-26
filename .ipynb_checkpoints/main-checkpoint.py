
import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata


automata = Cellular_automata(32,32,10,2)
a = automata.Grid()
plt.imshow(a)
plt.show()