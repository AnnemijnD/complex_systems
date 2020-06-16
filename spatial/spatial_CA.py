import numpy as np

SIZE = 100


def initialise():
    size = SIZE

    # create grids randomly

    # create habitat grid
    grid = np.random.randint(low=0,high=2,size=(size,size))

    # create allele grids
    grid_a = np.random.randint(low=0, high=2,size=(size,size))
    grid_b = np.random.randint(low=0, high=2,size=(size,size))

    return grid, grid_a, grid_b

def survival(grids):

    return grids

def mating(grids):

    return grids

def dispersal(grids):

    return grids


grid, grid_a, grid_b = initialise()

grids = [grid, grid_a, grid_b]
