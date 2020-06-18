import spatial_CA as model

"""
Run the model through model.run_model(iterations, size, survive, p, empty)

Necessary arguments:
- iterations

Optional arguments:
- size [int] (default is 50)
- survive [dict] (default is {1:0.8, 2:0.8})
- p [int] (default is 0.6)
- empty [int] (default is 0.2)
- grid_type [str] (default is RANDOM), options: RANDOM, STRUCTURED, NON_STRUCTURED
"""


model.run_model(5000, size=20,survive={1:0.8, 2:0.8}, p=0.85, empty=0, grid_type="STRUCTURED")
