import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.colors as mcolors
from pathlib import Path
import time
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

global ERROR
ERROR = 0.05

global R
R = 2

global grid_border
grid_border = np.zeros((SIZE, SIZE))




def neumann(mat, i, j,r):
    """
    Return neighbours from the neumann's neighbourhood.
    """
    neighbors = []
    neighbors_inds = []
    neighbors_rows = []
    neighbors_cols = []
    try:
        if not((i - r) < 0):
            if not -mat[i][j] == grid_border[i-r][j]:
                neighbors.append(mat[i-r][j])
                neighbors_inds.append((i-r, j))
                neighbors_rows.append(i-r)
                neighbors_cols.append(j)
    except:
        pass


    try:
        if not -mat[i][j] == grid_border[i][j-r]:
            neighbors.append(mat[i][j-r])
            neighbors_inds.append((i, j-r))
            neighbors_rows.append(i)
            neighbors_cols.append(j-r)
    except:
        pass


    try:
        if not -mat[i][j] == grid_border[i+r][j]:
            neighbors.append(mat[i+r][j])
            neighbors_inds.append((i+r, j))
            neighbors_rows.append(i+r)
            neighbors_cols.append(j)

    except:
        pass

    try:
        if not -mat[i][j] == grid_border[i][j+r]:
            neighbors.append(mat[i][j+r])
            neighbors_inds.append((i, j+r))
            neighbors_rows.append(i)
            neighbors_cols.append(j+r)
    except:
        pass


    return neighbors_inds, neighbors_rows, neighbors_cols

def rand_neumann_off(mat, i, j, offspring,r):
    """
    Return neighbours from the neumann's neighbourhood.
    Only if neighbor has offspring.
    """

    neighbors = []
    neighbors_inds = []

    try:
        if not((i - r) < 0):
            if not -mat[i][j] == grid_border[i-r][j]:
                if offspring[i-r][j] > 0:
                    neighbors.append(mat[i-r][j])
                    neighbors_inds.append((i-r, j))
    except:
        pass


    try:
        if offspring[i][j-r] > 0:
            if not -mat[i][j] == grid_border[i][j-r]:
                neighbors.append(mat[i][j-r])
                neighbors_inds.append((i, j-r))
    except:
        pass


    try:
        if offspring[i+r][j] > 0:
            if not -mat[i][j] == grid_border[i+r][j]:
                neighbors.append(mat[i+r][j])
                neighbors_inds.append((i+r, j))

    except:
        pass

    try:
        if offspring[i][j+r] > 0:
            if not -mat[i][j] == grid_border[i][j+r]:
                neighbors.append(mat[i][j+r])
                neighbors_inds.append((i, j+r))
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


    # create grids randomly

    # create habitat grid
    if type == "RANDOM":
        grid = np.random.randint(low=1,high=3,size=(SIZE, SIZE))


    elif type == "STRUCTURED":

        grid_a = np.zeros((SIZE, SIZE))
        grid_b = np.zeros((SIZE, SIZE))
        grid = np.zeros((SIZE, SIZE))


        if SIZE % 2 == 1:
            Hab1 = SIZE // 2
        else:
            Hab1 = SIZE // 2 - 1

        for row in range(SIZE):
            for col in range(SIZE):
                if row <= Hab1:
                    grid[row][col] = 1
                    if Hab1 - row < R:
                        grid_border[row][col] = -2
                    else:
                        grid_border[row][col] = None



                else:
                    grid[row][col] = 2
                    if row - R <= Hab1:
                        grid_border[row][col] = -1
                    else:
                        grid_border[row][col] = None





    elif type == "NON_STRUCTURED":
        try:
            grid = pickle.load(open(f"non_struct_habs/SIZE={SIZE}.p", "rb"))

        except:
            grid = np.random.randint(low=1,high=3,size=(SIZE, SIZE))
            pickle.dump(grid, open(f"non_struct_habs/SIZE={SIZE}.p", "wb"))

    # # create allele grids
    grid_a = np.random.randint(low=1, high=3,size=(SIZE, SIZE))
    grid_b = np.random.randint(low=1, high=3,size=(SIZE, SIZE))

    for i in range(len(grid_a)):
        for j in range(len(grid_a[0])):
            r = np.random.uniform(0, 1)
            if r < EMPTY_CELLS:
                grid_a[i][j] = 0
                grid_b[i][j] = 0



    return grid, grid_a, grid_b

def survival(grids):
    """
    Determines which individual on the grid survive the time step
    Args:
        grids (list): list of length 3 containing the 2d matrices with individuals

    Returns:
        grids (list) : list of length 3 containing the 2d matrices with individuals
    """

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
    """
    Creates offspring. Constructs an offspring matrix.

    Args:
        grids (list): list of length 3 containing the 2d matrices with individuals

    Returns:
        grids (list) : list of length 5 containing the 2d matrices with individuals
                        including offspring matrices

    """
    grid, grid_a, grid_b = grids

    mat_off_a = np.zeros((SIZE, SIZE))
    mat_off_b = np.zeros((SIZE, SIZE))

    for row in range(len(grid)):
        for col in range(len(grid)):
            inds, rows, cols = [], [], []
            for r in range(R+1):
                if r == 0:
                    continue
                neu = neumann(grid, row, col,r)
                inds += neu[0]
                rows += neu[1]
                cols += neu[2]

            offspring_a, offspring_b = mate(grids, inds, row, col)
            mat_off_a[row][col] = offspring_a
            mat_off_b[row][col] = offspring_b


    grids.append(mat_off_a)
    grids.append(mat_off_b)


    return grids


def mate(grids, inds, row, col):

    """
    Creates the offspring of one individual
    Args:
        grids (list) : list of length 3 containing the 2d matrices with individuals
        inds (list)  : the indices of the neighbors that can mate with the individual
        row (int)    : The row position of the individual
        col (int)    : The column position of the individual

    Returns:
        offspring_a (int) : Allele of type A that the offspring gets
        offspring_b (int) : Allele of type B that the offspring gets
    """
    S1ab, S1Ab, S1aB, S1AB, S2ab, S2Ab, S2aB, S2AB, S0 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    grid, grid_a, grid_b = grids

    inds.append((row, col))

    # Loop over variables to assign into mating pools
    for ind in inds:
        j = ind[0]
        i = ind[1]


        p = np.random.uniform(0, 1)
        if grid_a[j][i] == 0 and grid_b[j][i] == 0:
            S0 += 1

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

    p_matrix = np.array([probabilities(Sab, 1), probabilities(SaB, 2),
                        probabilities(SAb, 1), probabilities(SAB,2)])

    for i in range(len(p_matrix)):
        n = np.sum(p_matrix[i])
        if not n == 0:
            p_matrix[i] = p_matrix[i]/n


    # Loop over grid and create offspring matrix
    p = np.random.uniform(0, 1)


    offspring_a = 0
    offspring_b = 0
    if grid_a[row][col] == 0 and grid_b[row][col] == 0:
        offspring_a = 0
        offspring_b = 0

    elif grid_a[row][col] == 1 and grid_b[row][col] == 1:

        p1, p2, p3, p4 = p_matrix[0]
        if p < p1:
            offspring_a = 1
            offspring_b = 1
        elif p < p1 + p2:
            offspring_a = 2
            offspring_b = 1
        elif p < p1 + p2 + p3:
            offspring_a = 1
            offspring_b = 2
        else:
            offspring_a = 2
            offspring_b = 2

    elif grid_a[row][col] == 1 and grid_b[row][col] == 2:
        p1, p2, p3, p4 = p_matrix[1]
        if p < p1:
            offspring_a = 1
            offspring_b = 2
        elif p < p1 + p2:
            offspring_a = 2
            offspring_b = 2
        elif p < p1 + p2 + p3:
            offspring_a = 1
            offspring_b = 1
        else:
            offspring_a = 2
            offspring_b = 1

    elif grid_a[row][col] == 2 and grid_b[row][col] == 1:
        p1, p2, p3, p4 = p_matrix[2]
        if p < p1:
            offspring_a = 2
            offspring_b = 1
        elif p < p1 + p2:
            offspring_a = 1
            offspring_b = 1
        elif p < p1 + p2 + p3:
            offspring_a = 2
            offspring_b = 2
        else:
            offspring_a = 1
            offspring_b = 2

    elif grid_a[row][col] == 2 and grid_b[row][col] == 2:
        p1, p2, p3, p4 = p_matrix[3]
        if p < p1:
            offspring_a = 2
            offspring_b = 2
        elif p < p1 + p2:
            offspring_a = 1
            offspring_b = 2
        elif p < p1 + p2 + p3:
            offspring_a = 2
            offspring_b = 1
        else:
            offspring_a = 1
            offspring_b = 1

    return offspring_a, offspring_b
# Calculates probabilities
def probabilities(S, b):
    """
    Calculates the probabilites of creating a specific offspring type
    Args:
        S (list) : List of data to use in formula
        b (int)  : Reproducibility type
    """
    if S[0] == 0 and S[1] == 0:
        p1, p2, p3, p4 = 0, 0, 0, 0
    elif S[0] == 0 and S[1] > 0:
        if b == 1:
            p1 = ((1-MATING)/S[1]) * (S[6]+.5*S[7]+.5*S[8]+.25*S[9])
        else:
            p1 = ((MATING)/S[1]) * (S[6]+.5*S[7]+.5*S[8]+.25*S[9])

        if b == 1:
            p2 = ((1-MATING)/(2*S[1]))*(S[7]+.5*S[9])
        else:
            p2 = ((MATING)/(2*S[1]))*(S[7]+.5*S[9])

        if b == 1:
            p3 = ((1-MATING)/(2*S[1]))*(S[8]+.5*S[9])
        else:
            p3 = ((MATING)/(2*S[1]))*(S[8]+.5*S[9])

        p4 = 0

    elif S[1] == 0:
        if b == 1:
            p1 = (MATING/S[0]) * (S[2]+.5*S[3] + .5*S[4]+.25*S[5])
        else:
            p1 = ((1- MATING)/S[0]) * (S[2]+.5*S[3] + .5*S[4]+.25*S[5])

        if b == 1:
            p2 = (MATING/(2*S[0]))*(S[3]+.5*S[5])
        else:
            p2 = ((1 - MATING)/(2*S[0]))*(S[3]+.5*S[5])

        if b == 1:
            p3 = (MATING/(2*S[0]))*(S[4]+.5*S[5])
        else:
            p3 = ((1 - MATING)/(2*S[0]))*(S[4]+.5*S[5])


        if b == 1:
            p4 = (MATING/(4*S[0]))*(S[5]) + ((1-MATING)/(4*S[0]))*(S[9])
        else:
            p4 = ((1 - MATING)/(4*S[0]))*(S[5]) + ((MATING)/(4*S[0]))*(S[9])

    else:
        if b == 1:
            p1 = ((MATING/S[0]) * ( S[2]+.5*S[3] + .5*S[4]+.25*S[5])+((1-MATING)/S[1])
                  *(S[6]+.5*S[7]+.5*S[8]+.25*S[9]))
        else:
            p1 = (((1- MATING)/S[0]) * ( S[2]+.5*S[3] + .5*S[4]+.25*S[5])+((MATING)/S[1])
                  *(S[6]+.5*S[7]+.5*S[8]+.25*S[9]))

        if b == 1:
            p2 = (MATING/(2*S[0]))*(S[3]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[7]+.5*S[9])
        else:
            p2 = ((1 - MATING)/(2*S[0]))*(S[3]+.5*S[5])+((MATING)/(2*S[1]))*(S[7]+.5*S[9])


        if b == 1:
            p3 = (MATING/(2*S[0]))*(S[4]+.5*S[5])+((1-MATING)/(2*S[1]))*(S[8]+.5*S[9])
        else:
            p3 = ((1 - MATING)/(2*S[0]))*(S[4]+.5*S[5])+((MATING)/(2*S[1]))*(S[8]+.5*S[9])


        if b == 1:
            p4 = (MATING/(4*S[0]))*(S[5]) + ((1-MATING)/(4*S[0]))*(S[9])
        else:
            p4 = ((1 - MATING)/(4*S[0]))*(S[5]) + ((MATING)/(4*S[0]))*(S[9])

    return p1, p2, p3, p4

def dispersal(grids):
    """
    Places offspring in empty positions in the grid
    Args:
        grids (list) : list of length 5 with 2D arrays of individuals and
                        offspring matrices
    Returns:
        grids (list) : list of length 3 with 2D arrays of individuals

    """
    grid, grid_a, grid_b, offspring_a, offspring_b = grids

    for row in range(len(grid)):
        for col in range(len(grid[0])):

            # vind lege cel
            if grid_a[row][col] == 0:

                # get neighbors with offspring (von neumann)
                neighbors_inds = []
                for r in range(R+1):
                    if r == 0:
                        continue
                    neighbors_inds += rand_neumann_off(grid, row, col, offspring_a,r)



                # if only one neighbor, place in this cell
                if len(neighbors_inds) == 1:
                    neigh = neighbors_inds[0]
                    x = neigh[0]
                    y = neigh[1]

                    # place in cell
                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]

                    # remove offspring
                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


                # choose random offsrping
                elif not len(neighbors_inds) == 0:
                    rand_ind = random.randint(0, len(neighbors_inds) - 1)
                    chosen_neigh = neighbors_inds[rand_ind]
                    x = chosen_neigh[0]
                    y = chosen_neigh[1]

                    # place offspring in cell
                    grid_a[row][col] = offspring_a[x][y]
                    grid_b[row][col] = offspring_b[x][y]

                    # remove offspring
                    offspring_a[x][y] = 0
                    offspring_b[x][y] = 0


    grids = [grid, grid_a, grid_b]
    return grids

def make_figure(grids, plot=True):
    """
    Converts matrixs to grid with individual types 1-4, and can plot a heatmap based on
    these values
    Args:
        grids (list) : List of length 3 with 2D matrices of individuals
        plot (bool)  : Determines wether or not a heatmap is shown

    Returns:
        figure (2D matrix) : 2D matrix of individual grid with values ranging
                            from 1 to 4
    """
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
        norm = plt.Normalize(0,4)
        cmap = mcolors.LinearSegmentedColormap.from_list("n",['#FFFFFF','#20639B','#3CAEA3','#F6D55C','#ED553B'])
        sns.heatmap(figure, clim=(0, 4),cmap=cmap, norm=norm, vmin=0, vmax=4)
        plt.show()

    return figure


def linkage_diseq(counts):
    """
    Calculates linkage disequilibrium
    """
    N4, N3, N1, N2, N0 = counts
    ld = ((N0*N3)-(N1*N2)) / ((SIZE**2 - N4)**2)

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

    global grid_border
    grid_border = np.zeros((SIZE, SIZE))


    # Initialise grid
    grid, grid_a, grid_b = initialise(GRID_TYPE)

    grids = [grid, grid_a, grid_b]

    type_1 = []
    type_2 = []
    type_3 = []
    type_4 = []

    # holds all linkage diseq. vals
    ld_array = []
    i_s = 0
    figures = []
    prints = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,5000,
                5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000-1]
    for i in range(iterations):
        i_s +=1
        if i in prints:
            print(i)

        grids = survival(grids)

        grids = mating(grids)

        grids = dispersal(grids)

        grid, grid_a, grid_b = grids

        figure = make_figure(grids, plot=False)
        figures.append(figure)

        # keep up data for the plots
        unique, counts = np.unique(figure, return_counts=True)
        freqs = np.asarray((unique, counts)).T

        el_0 = 0
        el_1 = 0
        el_2 = 0
        el_3 = 0
        el_4 = 0

        for j in freqs:

            if j[0] == 0:
                el_0 = j[1]
            elif j[0] == 1:
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
        if el_0 == SIZE**2:
            break

        # calculate ld and add to array
        ld_counts = [el_0, el_1, el_2, el_3, el_4]
        ld = linkage_diseq(ld_counts)
        ld_array.append(ld)


    # make figure
    figure = make_figure(grids, plot=False)
    x = list(range(i_s))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # try:
    #     Path(f"/Users/annemijndijkhuis/Documents/Computational Science/Complex systems sims/complex_systems/spatial/data/allo/s={SURVIVAL}").mkdir(parents=True, exist_ok=True)
    #
    #     pickle.dump([x, type_1, type_2, type_3, type_4, ld_array, figures],
    #     open(f"data/allo/s={SURVIVAL}/n={iterations}_p={MATING}_{timestr}.p", "wb"))
    # except:
    #     pass

    return [x, type_1, type_2, type_3, type_4, ld_array, figures]
