import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
# import numba
import pickle
global SIZE
SIZE = 50

global SURVIVAL
SURVIVAL = {1:0.8, 2:0.8}

global MATING
MATING = 0.85

global EMPTY_CELLS
EMPTY_CELLS = 0.2

global GRID_TYPE
GRID_TYPE = "RANDOM"


def rand_neumann(mat, i, j, offspring):
    """
    Return a random neighbour from the neumann's neighbourhood.
    Only if neighbor has offspring.
    """

    neighbors = []
    neighbors_inds = []
    try:

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


def initialise(type="RANDOM"):
    """
    Initialises grid
    Args:
        type (str): RANDOM, STRUCTURED or NON_STRUCTURED

    Returns:
        grid (2D matrix): habitat grid
        grid_a (2D matrix): Allele a grid
        grid_b (2D matrix): Allele b grid

    """
    # size = SIZE

    # create grids randomly

    # create habitat grid
    if type == "RANDOM":
        grid = np.random.randint(low=1,high=3,size=(SIZE, SIZE))


    elif type == "STRUCTURED":

        print("SIZE",SIZE)
        grid = np.zeros((SIZE, SIZE))

        if SIZE % 2 == 1:
            Hab1 = SIZE // 2
        else:
            Hab1 = SIZE // 2 - 1

        for row in range(SIZE):
            for col in range(SIZE):
                if row <= Hab1:
                    grid[row][col] = 1
                else:
                    grid[row][col] = 2

    elif type == "NON_STRUCTURED":
        try:
            grid = pickle.load(open(f"non_struct_habs/SIZE={SIZE}.p", "rb"))

        except:
            grid = np.random.randint(low=1,high=3,size=(SIZE, SIZE))
            pickle.dump(grid, open(f"non_struct_habs/SIZE={SIZE}.p", "wb"))

    # create allele grids
    grid_a = np.random.randint(low=1, high=3,size=(SIZE, SIZE))
    grid_b = np.random.randint(low=1, high=3,size=(SIZE, SIZE))

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
            if grid_a[j][i] == 0 and grid_b[j][i] == 0:
                continue

            elif grid_a[j][i] == 1 and grid_b[j][i] == 1:
                if p < MATING:
                    S1ab += 1
                else:
                    S2ab += 1

            elif grid_a[j][i] == 1 and grid_b[j][i] == 2:
                if p < MATING:
                    S2aB += 1
                else:
                    S1aB += 1

            elif grid_a[j][i] == 2 and grid_b[j][i] == 1:
                if p < MATING:
                    S1Ab += 1
                else:
                    S2Ab += 1


            elif grid_a[j][i] == 2 and grid_b[j][i] == 2:
                if p < MATING:
                    S2AB += 1
                else:
                    S1AB += 1

    # Total mating pool sizes
    S1 = S1ab + S1Ab + S1aB + S1AB
    S2 = S2ab + S2Ab + S2aB + S2AB

    # Rearrange S variables to permute for the probabilities formula
    Sab = (S1, S2, S1ab, S1Ab, S1aB, S1AB, S2ab, S2Ab, S2aB, S2AB)
    SAb = (S1, S2, S1Ab, S1ab, S1AB, S1aB, S2Ab, S2ab, S2AB, S2aB)
    SaB = (S1, S2, S1aB, S1AB, S1ab, S1Ab, S2aB, S2AB, S2ab, S2Ab)
    SAB = (S1, S2, S1AB, S1aB, S1Ab, S1ab, S2AB, S2aB, S2Ab, S2ab)

    p_matrix = [probabilities(Sab), probabilities(SaB),
                        probabilities(SAb), probabilities(SAB)]

    # Loop over grid and create offspring matrix
    offspring_a = np.zeros((SIZE, SIZE))
    offspring_b = np.zeros((SIZE, SIZE))
    for i in range(len(grid)):
        for j in range(len(grid[0])):

            p = np.random.uniform(0, 1)
            if grid_a[j][i] == 0 & grid_b[j][i] == 0:
                offspring_a[j][i] = 0
                offspring_b[j][i] = 0

            elif grid_a[j][i] == 1 and grid_b[j][i] == 1:
                p1, p2, p3, p4 = p_matrix[0]
                if p < p1:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 1
                elif p < p1 + p2:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2
                elif p < p1 + p2 + p3:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 2
                else:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2

            elif grid_a[j][i] == 1 and grid_b[j][i] == 2:
                p1, p2, p3, p4 = p_matrix[1]
                if p < p1:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 2
                elif p < p1 + p2:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2
                elif p < p1 + p2 + p3:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 1
                else:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 1

            elif grid_a[j][i] == 2 and grid_b[j][i] == 1:
                p1, p2, p3, p4 = p_matrix[2]
                if p < p1:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2
                elif p < p1 + p2:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 1
                elif p < p1 + p2 + p3:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2
                else:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 2

            elif grid_a[j][i] == 2 and grid_b[j][i] == 2:
                p1, p2, p3, p4 = p_matrix[3]
                if p < p1:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 2
                elif p < p1 + p2:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 2
                elif p < p1 + p2 + p3:
                    offspring_a[j][i] = 2
                    offspring_b[j][i] = 1
                else:
                    offspring_a[j][i] = 1
                    offspring_b[j][i] = 1

    # offspring_a = np.array([[0,0,1], [0,0,2], [2,1,0]])
    # offspring_b = np.array([[0,0,1], [0,0,1], [1,1,0]])


    grids.append(offspring_a)
    grids.append(offspring_b)
    return grids, Sab

# Calculates probabilities
def probabilities(S):

    try:
        p1 = ((MATING/S[0])*(S[2]+.5*S[3]+.5*S[4]+.25*S[5])+((1-MATING)/S[1])
              *(S[6]+.5*S[7]+.5*S[8]+.25*S[9]))
    except:
        p1 = 0

    try:
        p2 = (MATING/(2*S[0]))*(S[3]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[7]+.5*S[9])
    except:
        p2 = 0

    try:
        p3 = (MATING/(2*S[0]))*(S[4]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[8]+.5*S[9])
    except:
        p3 = 0

    try:
        p4 = (MATING/(4*S[0]))*(S[5]) + ((1-MATING)/(4*S[1]))*(S[9])
    except:
        p4 = 0

    return p1, p2, p3, p4

def dispersal(grids):
    """
    Places offspring in empty positions in the grid
    """
    grid, grid_a, grid_b, offspring_a, offspring_b = grids

    for row in range(len(grid)):
        for col in range(len(grid[0])):

            # vind lege cel
            if grid_a[row][col] == 0:

                # get neighbors with offspring (von neumann)
                neighbors_inds = rand_neumann(grid_a, row, col, offspring_a)

                # als maar 1 neighbor: plaats deze in de cel
                if len(neighbors_inds) == 1:
                    neigh = neighbors_inds[0]
                    x = neigh[0]
                    y = neigh[1]

                    # plaats offspring in lege cel
                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]

                    # verwijder offspring
                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


                # kies random offspring van de neigbors
                elif not len(neighbors_inds) == 0:
                    rand_ind = random.randint(0, len(neighbors_inds) - 1)
                    chosen_neigh = neighbors_inds[rand_ind]
                    x = chosen_neigh[0]
                    y = chosen_neigh[1]

                    # plaats offspring in lege cel
                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]

                    # verwijder offspring
                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


    grids = [grid, grid_a, grid_b]
    return grids

def make_figure(grids, plot=True):

    grid, grid_a, grid_b = grids

    figure = np.zeros((SIZE, SIZE))
    for row in range(len(grid[0])):
        for col in range(len(grid[0])):
            if grid_a[row][col] == 1:

                if grid_b[row][col] == 1:
                    figure[row][col] = 1
                elif grid_b[row][col] == 2:
                    figure[row][col] = 2

            elif grid_a[row][col] == 2:
                if grid_b[row][col] == 1:
                    figure[row][col] = 3
                elif grid_b[row][col] == 2:
                    figure[row][col] = 4
    if plot:
        ax = sns.heatmap(figure)
        plt.show()

    return figure

# fake data
# grid = np.array([[1,1,1], [2,1,2], [2,2,2]])
# grid_a = np.array([[1,1,2], [0,0,2], [2,2,0]])
# grid_b = np.array([[2,2,1], [0,0,1], [2,1,0]])

def linkage_diseq(S):
    (S1, S2, S1ab, S1Ab, S1aB, S1AB, S2ab, S2Ab, S2aB, S2AB) = S
    N0 = S1ab + S2ab
    N1 = S1aB + S2aB
    N2 = S1Ab + S2Ab
    N3 = S1AB + S2AB
    L = S1 + S2

    ld = ((N0*N3) - (N1*N2)) / L**2

    return ld


def run_model(iterations, size=SIZE, survive=SURVIVAL, p=MATING, empty=EMPTY_CELLS, grid_type=GRID_TYPE):
# Redefine global variables when specified
    global SIZE
    SIZE = size

    global SURVIVAL
    SURVIVAL = survive

    global MATING
    MATING = p

    global EMPTY_CELLS
    EMPTY_CELLS = empty

    global GRID_TYPE
    GRID_TYPE = grid_type
    # Initialise grid
    grid, grid_a, grid_b = initialise(GRID_TYPE)
    grids = [grid, grid_a, grid_b]

    type_1 = []
    type_2 = []
    type_3 = []
    type_4 = []

    # holds all linkage diseq. vals
    ld_array = []

    prints = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    for i in range(iterations):
        if i in prints:
            print(i)
        grids = survival(grids)

        # let op, grids output hier is 5 elementen
        grids, S = mating(grids)

        # calculate ld and add to array
        ld = linkage_diseq(S)
        ld_array.append(ld)

        # en hier weer 3
        grids = dispersal(grids)

        grid, grid_a, grid_b = grids
        figure = make_figure(grids, plot=False)

        # keep up data for the plots
        unique, counts = np.unique(figure, return_counts=True)
        freqs = np.asarray((unique, counts)).T
        el_1 = 0
        el_2 = 0
        el_3 = 0
        el_4 = 0
        for j in freqs:

            if j[0] == 1:
                el_1 = j[1]
            elif j[0] == 2:
                el_2 = j[1]
            elif j[0] == 3:
                el_3 = j[1]
            elif j[0] == 4:
                el_4 = j[1]

        type_1.append(el_1)
        type_2.append(el_2)
        type_3.append(el_3)
        type_4.append(el_4)

    # mkake figure
    figure = make_figure(grids)
    x = list(range(iterations))


    # make freq plots
    plt.plot(x, type_1, label="ab")
    plt.plot(x, type_2, label="aB")
    plt.plot(x, type_3, label="Ab")
    plt.plot(x, type_4, label="AB")
    plt.legend()
    plt.show()

    plt.plot(ld_array)
    plt.show()

    return figure
