import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle

class Grid:
    def __init__(self, size, empty, survival, mating, type):
        self.grid = np.zeros((size, size), dtype=object_)

        if type == "RANDOM":
            habitats = [1, 2]
            types = [1, 2, 3, 4]
            for i in range(size):
                for j in range(size):
                    random.shuffle(habitats)
                    random.shuffle(types)
                    print(int(habitats[0]), int(types[0]))
                    self.grid[i][j] = Cell(int(habitats[0]), int(types[0]))

        elif type == "STRUCTURED":
            pass


class Cell:
    def __init__(self, H, type):
        self.H = H
        self.type = type

    def mating(self, mating):
        p = np.random.uniform(0, 1)
        if (self.type == 1 or self.type == 3) and p < mating:
            self.S = 2
        elif (self.type == 2 or self.type == 4) and p > mating:
            self.S = 2
        else:
            self.S = 1

grid = Grid(10, 0, 1, 1, "RANDOM")
