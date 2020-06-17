import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

SIZE = 5
SURVIVAL = {1:0.8, 2:0.8}
EMPTY_CELLS = 0.2


def rand_neumann(mat, i, j, offspring):
    """
    Return a random neighbour from the neumann's neighbourhood.
    Only if neighbor has offspring.
    """

    neighbors = []
    neighbors_inds = []
    try:
        if not ((i - 1) < 0):
            if offspring[i-1][j] > 0:
                neighbors.append(mat[i-1][j])
                neighbors_inds.append((i-1, j))
    except:
        pass


    try:
        if not((j - 1) < 0):
            if offspring[i][j-1] > 0:
                neighbors.append(mat[i][j-1])
                neighbors_inds.append((i, j-1))
    except:
        pass


    try:
        if offspring[i+1][j] > 0:
            neighbors.append(mat[i+1][j])
            neighbors_inds.append((i+1, j))

    except:
        pass

    try:
        if offspring[i][j+1] > 0:
            neighbors.append(mat[i][j+1])
            neighbors_inds.append((i, j+1))
    except:
        pass

    return neighbors_inds


def initialise():
    size = SIZE

    # create grids randomly

    # create habitat grid
    grid = np.random.randint(low=1,high=3,size=(size,size))

    # create allele grids
    grid_a = np.random.randint(low=1, high=3,size=(size,size))
    grid_b = np.random.randint(low=1, high=3,size=(size,size))

    # oeps toch een for loop
    for i in range(len(grid_a)):
        for j in range(len(grid_a[0])):
            r = np.random.uniform(0, 1)
            if r < EMPTY_CELLS:
                grid_a[i][j] = 0
                grid_b[i][j] = 0


    return grid, grid_a, grid_b

def survival(grids):
    grid, grid_a, grid_b = grids
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # if empty cell, continue
            if grid_a[i][j] == 0:
                continue

            # get chance
            if not grid[i][j] == grid_a[i][j]:
                chance = 1 - SURVIVAL[grid_a[i][j]]
            else:
                chance = SURVIVAL[grid_a[i][j]]

            # draw number
            r = np.random.uniform(0, 1)

            # death event
            if r > chance:
                grid_a[i][j] = 0
                grid_b[i][j] = 0

    grids = [grid, grid_a, grid_b]
    return grids

def mating(grids):

    offspring_a = np.array([[0,0,1], [0,0,2], [2,1,0]])
    offspring_b = np.array([[0,0,1], [0,0,1], [1,1,0]])
    grids.append(offspring_a)
    grids.append(offspring_b)
    return grids

def dispersal(grids):
    grid, grid_a, grid_b, offspring_a, offspring_b = grids

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid_a[row][col] == 0:

                # get neighbors with offspring (von neumann)
                neighbors_inds = rand_neumann(grid_a, row, col, offspring_a)

                if len(neighbors_inds) == 1:
                    neigh = neighbors_inds[0]
                    x = neigh[0]
                    y = neigh[1]

                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]
                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


                elif not len(neighbors_inds) == 0:
                    rand_ind = random.randint(0, len(neighbors_inds) - 1)
                    chosen_neigh = neighbors_inds[rand_ind]
                    x = chosen_neigh[0]
                    y = chosen_neigh[1]
                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]

                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


    grids = [grid, grid_a, grid_b]
    return grids


# grid, grid_a, grid_b = initialise()
grid = np.array([[1,1,1], [2,1,2], [2,2,2]])
grid_a = np.array([[1,1,2], [0,0,2], [2,2,0]])
grid_b = np.array([[2,2,1], [0,0,1], [2,1,0]])
grids = [grid, grid_a, grid_b]

grids = mating(grids)

grids = dispersal(grids)


grid, grid_a, grid_b = grids

figure = np.zeros((3, 3))
for row in range(len(grid[0])):
    for col in range(len(grid[0])):
        if grid_a[row][col] == 1:
            if grid_b[row][col] == 1:
                figure[row][col] = 1
            else:
                figure[row][col] = 2
        else:
            if grid_b[row][col] == 1:
                figure[row][col] = 3
            else:
                figure[row][col] = 4
ax = sns.heatmap(figure)
plt.show()
