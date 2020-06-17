import numpy as np

SIZE = 10
SURVIVAL = {1:0.8, 2:0.8}
MATING = 0.6
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
    S1ab, S1Ab, S1aB, S1AB, S2ab, S2Ab, S2aB, S2AB = 0, 0, 0, 0, 0, 0, 0, 0
    grid, grid_a, grid_b = grids

    # Loop over variables to assign into mating pools
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            p = np.random.uniform(0, 1)
            if grid_a[j][i] == 0 & grid_b[j][i] == 0:
                continue

            elif grid_a[j][i] == 1 & grid_b[j][i] == 1:
                if p < MATING:
                    S1ab += 1
                else:
                    S2ab += 1

            elif grid_a[j][i] == 1 & grid_b[j][i] == 2:
                if p < MATING:
                    S2aB += 1
                else:
                    S1aB += 1

            elif grid_a[j][i] == 2 & grid_b[j][i] == 1:
                if p < MATING:
                    S1Ab += 1
                else:
                    S2Ab += 1

            elif grid_a[j][i] == 2 & grid_b[j][i] == 2:
                if p < MATING:
                    S2AB += 1
                else:
                    S1AB += 1

    # Total mating pool sizes
    S1 = S1ab + S1Ab + S1aB + S1AB
    S2 = S2ab + S2Ab + S2aB + S2AB

    # Rearrange S variables to permutefor  the probabilities formula
    Sab = (S1, S2, S1ab, S1Ab, S1aB, S1AB, S2ab, S2Ab, S2aB, S2AB)
    SAb = (S1, S2, S1Ab, S1ab, S1AB, S1aB, S2Ab, S2ab, S2AB, S2aB)
    SaB = (S1, S2, S1aB, S1AB, S1ab, S1Ab, S2aB, S2AB, S2ab, S2Ab)
    SAB = (S1, S2, S1AB, S1aB, S1Ab, S1ab, S2AB, S2aB, S2Ab, S2ab)

    p_matrix = np.array([[probabilities(Sab)], [probabilities(SaB)],
                        [probabilities(SAb)], [probabilities(SAB)]])

    return p_matrix

# Calculates probabilities
def probabilities(S):

    p1 = ((MATING/S[0])*(S[2]+.5*S[3]+.5*S[4]+.25*S[5])+((1-MATING)/S[1])
          *(S[6]+.5*S[7]+.5*S[8]+.25*S[9]))
    p2 = (MATING/(2*S[0]))*(S[3]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[7]+.5*S[9])
    p3 = (MATING/(2*S[0]))*(S[4]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[8]+.5*S[9])
    p4 = (MATING/(4*S[0]))*(S[5]) + ((1-MATING)/(4*S[1]))*(S[9])

    return p1, p2, p3, p4

def dispersal(grids):

    return grids


grid, grid_a, grid_b = initialise()

grids = [grid, grid_a, grid_b]
grids = survival(grids)

print(mating(grids))
