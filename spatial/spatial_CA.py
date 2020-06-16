import numpy as np

SIZE = 10
SURVIVAL = {1:0.8, 2:0.8}
EMPTY_CELLS = 0.2


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

    return grids

def dispersal(grids):

    return grids


grid, grid_a, grid_b = initialise()

grids = [grid, grid_a, grid_b]
grids = survival(grids)
