import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10*np.pi,10*np.pi,200)

fig = plt.figure(figsize = (12,10))
def f(x):
	return np.sin(np.pi*x/20.)

noise = 0.6*np.random.normal(0, 1, len(x)) 
y = f(x)+noise
plt.plot(x,y,"b-",alpha = 0.5)
plt.fill_between(x,y,-10,alpha = 0.05)
plt.axis("off")
plt.ylim(-10,10)
plt.savefig("figure1.png",transparent = True)
plt.show()