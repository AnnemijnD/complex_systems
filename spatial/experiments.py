import spatial_CA as model
import numpy as np

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



<<<<<<< HEAD
figure = model.run_model(1000, size=50, empty=0.1, survive={1:0.55, 2:0.55}, p=1, grid_type="STRUCTURED")
=======
# model.run_model(20000, size=50,survive={1:0.8, 2:0.8}, p=1, empty=0.2, grid_type="STRUCTURED")
model.run_model(6000, size=20, grid_type="RANDOM", p=0.85, survive={1:0.8, 2:0.8}, empty=0)
>>>>>>> fec34b3de67509173340c6dd00e815e57a6cf5a5
