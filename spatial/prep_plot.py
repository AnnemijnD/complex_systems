import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.colors as mcolors

import glob

ERROR = 0.0005
#
ss = [{1: 1.0, 2: 1.0}, {1:0.99, 2:0.99}, {1:0.98, 2:0.98}, {1:0.97, 2:0.97},
{1:0.96, 2:0.96}, {1:0.95, 2:0.95}, {1:0.94, 2:0.94}, {1:0.93, 2:0.93}, {1:0.92, 2:0.92},
{1:0.91, 2:0.91}, {1:0.9, 2:0.9},{1:0.8, 2:0.8}, {1:0.7, 2:0.7},
{1:0.6, 2:0.6}, {1:0.5, 2:0.5}]

divers = {}
genpol = {}
"/Volumes/DATA/CompSys"
for s in ss:
    files = glob.glob(f"/Volumes/DATA/CompSys/allo/s={s}/n=10000*")

    if len(glob.glob(f"/Volumes/STAGE/CompSys/allo/s={s}/n=10000*")) > 0:
        files += glob.glob(f"/Volumes/STAGE/CompSys/allo/s={s}/n=10000*")

    print(files)

    # files = glob.glob(f"data/allo/s={s}/n=10000*")


    for file in files:

        print(file)
        try:
            list = pickle.load(open(file, "rb"))
        except:
            continue


        x, type_1, type_2, type_3, type_4, ld_array, figures = list

        if len(figures[0]) % 2 == 1:
            Hab1 = len(figures[0]) // 2
        else:
            Hab1 = len(figures[0]) // 2 - 1

        H1as = []
        H1As = []
        H2as = []
        H2As = []

        count = 0
        plot = True

        for figure in figures:
            if count == len(ld_array):
                break
            if abs(ld_array[count]) >= 0.25 - ERROR:

                if s[1] in divers:
                    divers[s[1]].append(count)
                else:
                    divers[s[1]] = [count]
                break

                # genetic polymorphism
            elif abs(ld_array[count]) == 0:

                genpoltrue = True
                for i in range(20):
                    if count - i < 0:
                        break
                    j = count - i
                    if not ld_array[j] == 0:
                        genpoltrue = False
                        break

                if genpoltrue:
                    # norm = plt.Normalize(0,4)
                    # cmap = mcolors.LinearSegmentedColormap.from_list("n",['#FFFFFF','#20639B','#3CAEA3','#F6D55C','#ED553B'])
                    # sns.heatmap(figure, clim=(0, 4),cmap=cmap, norm=norm, vmin=0, vmax=4)
                    # plt.show()
                    # plt.plot(ld_array)
                    # plt.show()

                    if s[1] in genpol:
                        genpol[s[1]].append(count)
                    else:
                        genpol[s[1]] = [count]
                    break



            # el_1a = 0
            # el_2a = 0
            # el_1A = 0
            # el_2A = 0
            # for row in range(len(figure)):
            #     for col in range(len(figure[0])):
            #                 if row <= Hab1:
            #                     if figure[row][col] == 1 or figure[row][col] == 2:
            #                         el_1a += 1
            #                     elif figure[row][col] == 3 or figure[row][col] == 4:
            #                         el_1A += 1
            #
            #                 else:
            #                     if figure[row][col] == 1 or figure[row][col] == 2:
            #                         el_2a += 1
            #                     elif figure[row][col] == 3 or figure[row][col] == 4:
            #                         el_2A += 1
            #
            #
            # if el_1A == 0 and el_2a == 0:
            #     if plot == True:
            #         print("divers", count)
            #         plot = False
            #
            #     if s[1] in divers:
            #         divers[s[1]].append(count)
            #     else:
            #         divers[s[1]] = [count]
            #
            #
            #
            # H1as.append(el_1a)
            # H1As.append(el_1A)
            # H2as.append(el_2a)
            # H2As.append(el_2A)
            count+=1

        #
        # norm = plt.Normalize(0,4)
        # cmap = mcolors.LinearSegmentedColormap.from_list("n",['#FFFFFF','#20639B','#3CAEA3','#F6D55C','#ED553B'])
        # sns.heatmap(figures[-1], clim=(0, 4),cmap=cmap, norm=norm, vmin=0, vmax=4)
        # plt.show()
        # plt.plot(ld_array)
        # plt.show()
        #
        # #
        # plt.figure()
        #
        # plt.subplot(121)
        # plt.plot(np.arange(count), H1as, label="a")
        # plt.plot(np.arange(count), H1As, label="A")
        # plt.title("1")
        # plt.legend()
        # # sp2
        # plt.subplot(122)
        # plt.plot(np.arange(count), H2as, label="a")
        # plt.plot(np.arange(count), H2As, label="A")
        # plt.title("2")
        # plt.legend()
        #
        # plt.show()
        # end = input()
        # if end == "True" or end == True:
        #     exit()
        # elif end == "break":
        #     break

    print(divers)
    print(genpol)

pickle.dump(divers, open(f"data/plot/divers.p", "wb"))
pickle.dump(genpol, open(f"data/plot/genpol.p", "wb"))

means_pol = []
stdevs_pol = []

means_spec = []
stdevs_spec = []

xaxis = []
for key in genpol.keys():

    mean1 = np.mean(genpol[key])
    stdev1 = np.std(genpol[key])
    xaxis.append(key)


    means_pol.append(mean1)
    stdevs_pol.append(stdev1)

    mean2 = np.mean(divers[key])
    stdev2 = np.std(divers[key])

    means_spec.append(mean2)
    stdevs_spec.append(stdev2)
