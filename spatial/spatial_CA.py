import numpy as np

SIZE = 100
SURVIVAL = {0:0.8, 1:0.8}
EMPTY_CELLS = 0.2


def initialise():
    size = SIZE

    # create grids randomly

    # create habitat grid
    grid = np.random.randint(low=0,high=2,size=(size,size))

    # create allele grids
    grid_a = np.random.randint(low=1, high=3,size=(size,size))
    grid_b = np.random.randint(low=1, high=3,size=(size,size))

    # oeps toch een for loop
    for i in range(len(grid_a)):
        for j in range(len(grid_a[0])):
            r = np.random.uniform(0, 1)
            if r > 0.8:
                grid_a[i][j] = 0
                grid_b[i][j] = 0


    return grid, grid_a, grid_b

def survival(grids):

    return grids

def mating(grids):

    return grids

def dispersal(grids):

    return grids


grid, grid_a, grid_b = initialise()
print(grid_a[0])
print(grid_b[0])
grids = [grid, grid_a, grid_b]
