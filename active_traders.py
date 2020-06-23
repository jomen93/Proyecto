import matplotlib.pyplot as plt 
from automata_1 import Cellular_automata
import numpy as np
from scipy.stats import powerlaw
size = 128
A = 1.0
a = 0.001
h = 0
steps = 512


p1 = 0.2; p2 = 0.1; p3 = 0.15; p4 = 0.2; p5 = 0.25

system_p1 = Cellular_automata(size,A,a,h)
# system_p2 = Cellular_automata(size,A,a,h)
# system_p3 = Cellular_automata(size,A,a,h)
# system_p4 = Cellular_automata(size,A,a,h)
# system_p5 = Cellular_automata(size,A,a,h)

system_p1.make_grid(p1,p1)
# system_p2.make_grid(p2,p2)
# system_p3.make_grid(p3,p3)
# system_p4.make_grid(p4,p4)
# system_p5.make_grid(p5,p5)

system_p1.make_clusters()
# system_p2.make_clusters()
# system_p3.make_clusters()
# system_p4.make_clusters()
# system_p5.make_clusters()

active_p1 = []
# active_p2 = []
# active_p3 = []
# active_p4 = []
# active_p5 = []

size_cluster = []

for i in range(steps):
	print("progress = \033[91m {:.2f}%\033[0m".format((i/steps)*100)+"\r",end ="")
	
	system_p1.make_clusters()
	# system_p2.make_clusters()
	# system_p3.make_clusters()
	# system_p4.make_clusters()
	# system_p5.make_clusters()

	system_p1.update_grid_ising()
	# system_p2.update_grid_ising()
	# system_p3.update_grid_ising()
	# system_p4.update_grid_ising()
	# system_p5.update_grid_ising()	
	
	system_p1.Random_variables()
	# system_p2.Random_variables()
	# system_p3.Random_variables()
	# system_p4.Random_variables()
	# system_p5.Random_variables()

	# system_p1.update_traders()
	# system_p2.update_traders()
	# system_p3.update_traders()
	# system_p4.update_traders()
	# system_p5.update_traders()
	
	active_p1.append(np.sum(system_p1.grid>0)+np.sum(system_p1.grid<0))
	# active_p2.append(np.sum(system_p2.grid>0)+np.sum(system_p2.grid<0))
	# active_p3.append(np.sum(system_p3.grid>0)+np.sum(system_p3.grid<0))
	# active_p4.append(np.sum(system_p4.grid>0)+np.sum(system_p4.grid<0))
	# active_p5.append(np.sum(system_p5.grid>0)+np.sum(system_p5.grid<0))
	for k in range(system_p1.n_clusters):
		size_cluster.append(len(list(np.argwhere(system_p1.grid_label == k))))
		
# plt.plot(size_cluster,"b.",label = "p = "+str(p1))
# plt.plot(active_p2,"r-",label = "p = "+str(p2))
# plt.plot(active_p3,"g-",label = "p = "+str(p3))
# plt.plot(active_p4,"k-",label = "p = "+str(p4))
# plt.plot(active_p5,"b:",label = "p = "+str(p5))
# plt.hist(size_cluster, color='#0504aa',alpha=0.7, rwidth=0.85)


y, x = np.histogram(size_cluster, bins = len(size_cluster))
np.savetxt('x.csv', x, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')

