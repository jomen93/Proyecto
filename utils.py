import numpy as np 
import matplotlib.pyplot as plt 

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
	axs[0, 0].plot(days, Open_price)
	axs[0, 0].set_title('Open Price')


	axs[0, 1].plot(days, High)
	axs[0, 1].set_title('High price')

	axs[0, 2].plot(days, Low)
	axs[0, 2].set_title('Low price')

	axs[1, 0].plot(days, Close)
	axs[1, 0].set_title('Close price')

	axs[1, 1].plot(days, Adj_close)
	axs[1, 1].set_title('Adjoint close')

	axs[1, 2].plot(days, Volume)
	axs[1, 2].set_title('Volume operation')

	for ax in axs.flat:
	    ax.set(xlabel='days', ylabel='price')

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
	    ax.label_outer()
	    ax.grid(True)

	plt.savefig("Data_complete")
	print("plot save!")