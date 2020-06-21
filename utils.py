import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.patches as mpatches


def plot_all(data):
	days = np.arange(np.shape(data)[0])
	Open_price = data[:,0]
	High = data[:,1]
	Low = data[:,2]
	Close = data[:,3]
	Adj_close = data[:,4]
	Volume = data[:,5]


	fig, axs = plt.subplots(2, 3, figsize = (15,10))
	x = 1;y=1
	axs[0, 0].plot(days, Open_price,"b-")
	axs[0, 0].set_title('Open Price')


	axs[0, 1].plot(days, High,"b-")
	axs[0, 1].set_title('High price')

	axs[0, 2].plot(days, Low,"b-")
	axs[0, 2].set_title('Low price')

	axs[1, 0].plot(days, Close,"b-")
	axs[1, 0].set_title('Close price')

	axs[1, 1].plot(days, Adj_close,"b-")
	axs[1, 1].set_title('Adjoint close')

	axs[1, 2].plot(days, Volume,"b-")
	axs[1, 2].set_title('Volume operation')

	for ax in axs.flat:
	    ax.set(xlabel='days', ylabel='price')

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
	    ax.label_outer()
	    ax.grid(True)

	plt.savefig("Data_complete", transparent =True)
	print("plot save!")

def logarithmic_price(data, Normalized = True):
	R = np.roll(data,-1) - data
	R = R[:-1]

	if Normalized == True:
		R = (R-np.mean(R))/np.std(R)
		return R
	else:
		return R





