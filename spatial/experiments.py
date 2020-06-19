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



# model.run_model(20000, size=50,survive={1:0.8, 2:0.8}, p=1, empty=0.2, grid_type="STRUCTURED")
model.run_model(10000, size=50, grid_type="STRUCTURED", p=1, survive={1:0.6, 2:0.6}, empty=0)
