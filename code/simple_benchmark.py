"""
Run simple benchmark
"""

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

# Define mesh
mesh_shape = tuple(100 for _ in range(3))
mesh_spacings = (10, 10, 5)
mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
n_active_cells = np.sum(active_cells)
susceptibility = create_susceptibilty(n_active_cells)

# Define receivers and survey
height = 100
shape = tuple(120 for _ in range(2))
grid_coords = create_observation_points(get_region(mesh), shape, height)
survey = create_survey(grid_coords)
model_map = maps.IdentityMap(nP=susceptibility.size)

# Run benchmark
benchmarker = SimulationBenchmarker(
    n_runs=1,
    survey=survey,
    mesh=mesh,
    ind_active=active_cells,
    chiMap=model_map,
    engine="geoana",
    store_sensitivities="ram",
    n_processes=None,
)
runtime, std = benchmarker.benchmark(susceptibility)
