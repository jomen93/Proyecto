import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

grid = np.random.choice([-1,1,0], size = (64,64),p=[0.3,0.3,0.4])

# grid[40, 40] = 1
# grid[40, 41] = 1
# grid[40, 42] = 1
# grid[41, 42] = 1
# grid[42, 40] = 1
# grid[42, 41] = 1
# grid[42, 42] = 1

def count_neighbors(grid,i,j):
	aux_grid = grid[i-1:i+2,j-1:j+2]
	if grid[i,j] !=0:
		return (np.count_nonzero(aux_grid)-1)
	else:
		return (np.count_nonzero(aux_grid))

def step_sell(grid):
	for i in range(1,len(grid)-1):
		for j in range(1,len(grid)-1):
			# Cualquier célula viva con menos de dos vecinos vivos muere, como por subpoblación.
			if grid[i,j] == 1 and count_neighbors(grid,i,j) < 2:
				grid[i,j] = 0
			# Cualquier célula viva con dos o tres vecinos vivos vive hasta la próxima generación.
			if grid[i,j] == 1 and count_neighbors(grid,i,j) ==2 and count_neighbors(grid,i,j) ==3:
				grid[i,j] = 1
			# Cualquier célula viva con más de tres vecinos vivos muere, como por sobrepoblación.
			if grid[i,j] == 1 and count_neighbors(grid,i,j) >3:
				grid[i,j] = 0
			# Cualquier célula muerta con exactamente tres vecinos vivos se convierte en una célula viva, como por reproducción.
			if grid[i,j] == 0 and count_neighbors(grid,i,j) ==3:
				grid[i,j] = 1

def step_buy(grid):
	for i in range(1,len(grid)-1):
		for j in range(1,len(grid)-1):
			# Cualquier célula viva con menos de dos vecinos vivos muere, como por subpoblación.
			if grid[i,j] == -1 and count_neighbors(grid,i,j) < 2:
				grid[i,j] = 0
			# Cualquier célula viva con dos o tres vecinos vivos vive hasta la próxima generación.
			if grid[i,j] == -1 and count_neighbors(grid,i,j) ==2 and count_neighbors(grid,i,j) ==3:
				grid[i,j] = -1
			# Cualquier célula viva con más de tres vecinos vivos muere, como por sobrepoblación.
			if grid[i,j] == -1 and count_neighbors(grid,i,j) >3:
				grid[i,j] = 0
			# Cualquier célula muerta con exactamente tres vecinos vivos se convierte en una célula viva, como por reproducción.
			if grid[i,j] == 0 and count_neighbors(grid,i,j) ==3:
				grid[i,j] = -1

ims = []
ims.append([plt.imshow(grid)])
fig = plt.figure()
for _ in range(100):
	step_sell(grid)
	step_buy(grid)
	im = plt.imshow(grid,animated=True)
	ims.append([im])

ani = animation.ArtistAnimation(fig,ims,interval = 200)
plt.show()


