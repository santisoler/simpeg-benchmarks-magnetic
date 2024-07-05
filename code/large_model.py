"""
Run simple benchmark
"""

import sys
import numpy as np
from simpeg import maps

from utilities import (
    get_region,
    create_observation_points,
    create_survey,
    create_susceptibilty,
    create_tensor_mesh,
    SimulationBenchmarker,
)

if len(sys.argv) > 2:
    raise IOError(f"Found multiple engine arguments: '{sys.argv[1:]}'. Pass only one.")

engine = sys.argv[1]
valid_engines = ("geoana", "choclo", "dask")
if engine not in valid_engines:
    raise ValueError(f"Invalid engine '{engine}'.")

# Define mesh
n = 100
mesh_shape = (n, n, n)
mesh_spacings = (10, 10, 5)
mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
n_active_cells = np.sum(active_cells)
susceptibility = create_susceptibilty(n_active_cells)

# Define receivers and survey
height = 100
m = 90
shape = (m, m)
grid_coords = create_observation_points(get_region(mesh), shape, height)
survey = create_survey(grid_coords, components="tmi")
model_map = maps.IdentityMap(nP=susceptibility.size)

# Run benchmark
if engine == "dask":
    engine = "geoana"
    import simpeg.dask  # noqa: F401
if engine == "geoana":
    kwargs = dict(n_processes=None)
else:
    kwargs = dict(numba_parallel=True)

benchmarker = SimulationBenchmarker(
    n_runs=1,
    survey=survey,
    mesh=mesh,
    ind_active=active_cells,
    chiMap=model_map,
    engine=engine,
    store_sensitivities="ram",
    **kwargs,
)
runtime, std = benchmarker.benchmark(susceptibility)
