import spatial_CA as model
import spatial_CA_allopatric as allo_model
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Run the model through model.run_model(iterations, size, survive, p, empty, grid_type)

Necessary arguments:
- iterations

Optional arguments:
- size [int] (default is 50)
- survive [dict] (default is {1:0.8, 2:0.8})
- p [int] (default is 0.6)
- empty [int] (default is 0.2)
- grid_type [str] (default is RANDOM), options: RANDOM, STRUCTURED, NON_STRUCTURED
"""

if len(sys.argv) != 3:
    print("Please enter 2 arguments: MODEL (SYMPATRIC, ALLOPATRIC), TYPE(STANDARD, CUSTOM)")
    exit()
else:
    if sys.argv[1] not in ["SYMPATRIC", "ALLOPATRIC"] or sys.argv[2] not in ["STANDARD", "CUSTOM"]:
        print("Please enter 2 arguments: MODEL (SYMPATRIC, ALLOPATRIC), TYPE(STANDARD, CUSTOM)")
    else:
        if sys.argv[1] == "SYMPATRIC" and sys.argv[2] == "STANDARD":
            print("Estimated Total Runtime: ~ 6min.")
            print("Running model 1/5: 3 figures: 10000 iterations")
            model.run_model(10000, size=25, empty=0, survive={1:0.8, 2:0.8}, p=0.85, grid_type="STRUCTURED")
            print("Running model 2/5: 3 figures: 10000 iterations")
            model.run_model(10000, size=25, empty=0, survive={1:0.85, 2:0.85}, p=1, grid_type="RANDOM")
            print("Running model 3/5: 3 figures: 10000 iterations")
            model.run_model(10000, size=25, empty=0, survive={1:0.7, 2:0.7}, p=1, grid_type="STRUCTURED")
            print("Running model 4/5: 3 figures: 10000 iterations")
            model.run_model(10000, size=25, empty=0, survive={1:0.6, 2:0.6}, p=1, grid_type="RANDOM")
            print("Running model 5/5: 1 figure: 20 x max 5000 iterations (but probably much less)")
            times = []
            ps = []
            for x in np.arange(0.6, 1.0, 0.1):
                time = []
                for z in range(5):
                    print(f"s = {x}, run {z+1}")
                    time.append(model.run_model(5000, size=25, empty=0, survive={1:x, 2:x}, p=1, grid_type="STRUCTURED", plot=False, break_at_speciation=True))
                times.append(np.average(time))
                ps.append(x)

            # Plot
            plt.scatter(ps, times)
            plt.xlabel("survival s")
            plt.ylabel("iteration")
            plt.title("Speciation time (|LD - eps| > 0.245)")
            plt.show()

        elif sys.argv[1] == "SYMPATRIC" and sys.argv[2] == "CUSTOM":
            while True:
                print("ENTER GRID SIZE (STANDARD is 25)")
                try:
                    size = int(input())
                    break
                except:
                    print("INVALID INPUT")

            while True:
                print("ENTER # OF ITERATIONS (STANDARD is 5000)")
                try:
                    i = int(input())
                    break
                except:
                    print("INVALID INPUT")

            while True:
                print("ENTER SURVIVAL PARAMETER S (0.5 < S < 1.0)")
                try:
                    x = float(input())
                    if x >= 0.5 and x <= 1.0:
                        break
                except:
                    print("INVALID INPUT")

            while True:
                print("ENTER MATING PARAMETER P (0.5 < S < 1.0)")
                try:
                    p = float(input())
                    if p >= 0.5 and p <= 1.0:
                        break
                except:
                    print("INVALID INPUT")

            while True:
                print("ENTER GRID_TYPE (RANDOM OR STRUCTURED)")
                grid_type = input()
                grid_type = grid_type.upper()
                if grid_type in ["RANDOM", "STRUCTURED"]:
                    break
                else:
                    print("INVALID INPUT")

            print("Running custom model...")
            model.run_model(i, size=size, empty=0, survive={1:x, 2:x}, p=p, grid_type=grid_type, plot=True)


        elif sys.argv[1] == "SYMPATRIC" and sys.argv[2] == "CUSTOM":
            # Stnadaard allopatrisch resultaten
            pass

        elif sys.argv[1] == "SYMPATRIC" and sys.argv[2] == "CUSTOM":
            # Custom sympatrisch resultaten
            pass



# model.run_model(1000, size=50,survive={1:0.8, 2:0.8}, p=1, empty=0.2, grid_type="STRUCTURED")
# model.run_model(20000, size=50, grid_type="NON_STRUCTURED", p=0.8, survive={1:0.85, 2:0.85}, empty=0)
# model.run_model(6000, size=20, grid_type="RANDOM", p=0.85, survive={1:0.8, 2:0.8}, empty=0)
# allo_model.run_model(1000, size=20, grid_type="STRUCTURED", empty=0, survive={1:0.95, 2:0.95}, p=0.5)
