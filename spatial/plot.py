import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.colors as mcolors

def plot1():
    genpol = pickle.load(open(f"data/plot/genpol.p", "rb"))
    divers = pickle.load(open(f"data/plot/divers.p", "rb"))


    means_pol = []
    stdevs_pol = []

    means_spec = []
    stdevs_spec = []

    xaxis = []
    xaxis2 = []
    for key in genpol.keys():

        mean1 = np.mean(genpol[key])
        stdev1 = np.std(genpol[key])
        xaxis.append(key)


        means_pol.append(mean1)
        stdevs_pol.append(stdev1)


    for key in divers.keys():
        xaxis2.append(key)
        mean2 = np.mean(divers[key])
        stdev2 = np.std(divers[key])

        means_spec.append(mean2)
        stdevs_spec.append(stdev2)

    ss = [{1: 1.0, 2: 1.0}, {1:0.99, 2:0.99}, {1:0.98, 2:0.98}, {1:0.97, 2:0.97},
    {1:0.96, 2:0.96}]

    ss1 = [{1:0.95, 2:0.95}, {1:0.94, 2:0.94}, {1:0.93, 2:0.93}, {1:0.92, 2:0.92},
    {1:0.91, 2:0.91}, {1:0.9, 2:0.9},{1:0.8, 2:0.8}, {1:0.7, 2:0.7},
    {1:0.6, 2:0.6}, {1:0.5, 2:0.5}]

    # for s in ss:
    #
    #     if s[1] not in genpol:
    #         means_pol.append(10000)
    #         stdevs_pol.append(0)
    #         xaxis.append(s[1])
    #
    #     if s[1] not in divers:
    #         means_spec.append(10000)
    #         stdevs_spec.append(0)
    #         xaxis2.append(s[1])


    # plt.figure()
    #
    # plt.subplot(121)
    # plt.errorbar(xaxis, means_pol, yerr=stdevs_pol, fmt="o")
    # plt.xlabel("s")
    # plt.ylabel("Generations")
    # plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.title("Genetic Polymorphism time")
    #
    # # for s in ss:
    # #
    # #     if s[1] not in genpol:
    # #         plt.scatter([s[1]], [10000], color="red")
    #
    #
    # plt.subplot(122)
    plt.errorbar(xaxis2, means_spec, yerr=stdevs_spec, fmt="o")
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel("s")
    plt.title("Speciation time")
    plt.ylabel("Generations")

    # for s in ss:
    #
    #     if s[1] not in divers:
    #         plt.scatter([s[1]], [10000], color="red")



    plt.show()

    return 0

def plot2():
    hab1 = pickle.load(open(f"non_struct_habs/SIZE=50.p", "rb"))
    SIZE = 50

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

            else:
                grid[row][col] = 2


    norm = plt.Normalize(1,2)
    cmap = mcolors.LinearSegmentedColormap.from_list("n",['#00203FFF', '#ADEFD1FF'])
    sns.heatmap(grid, clim=(1, 2),cmap=cmap, norm=norm, vmin=1, vmax=2)
    plt.show()

    sns.heatmap(hab1, clim=(1, 2),cmap=cmap, norm=norm, vmin=1, vmax=2)
    plt.show()

    return 0

def plot3():

    grids = pickle.load(open("data/allo/s={1: 0.5, 2: 0.5}/n=3_p=0.5_20200623-134816.p", "rb"))
    grid = grids[-1][0]

    norm = plt.Normalize(0,4)
    cmap = mcolors.LinearSegmentedColormap.from_list("n",['#FFFFFF','#20639B','#3CAEA3','#F6D55C','#ED553B'])
    sns.heatmap(grid, clim=(0, 4),cmap=cmap, norm=norm, vmin=0, vmax=4)
    plt.show()
    return 0
plot3()
