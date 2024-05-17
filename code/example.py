import itertools
import numpy as np
from simpeg import maps

from utilities import (
    get_region,
    create_dataset,
    create_observation_points,
    create_survey,
    create_susceptibilty,
    create_tensor_mesh,
    SimulationBenchmarker,
)

# Define mesh
mesh_shape = (30, 30, 30)
mesh_spacings = (10, 10, 5)
mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
n_active_cells = np.sum(active_cells)
susceptibility = create_susceptibilty(n_active_cells)

# Define receivers and survey
height = 100
shape = (40, 40)
grid_coords = create_observation_points(get_region(mesh), shape, height)
survey = create_survey(grid_coords)
model_map = maps.IdentityMap(nP=susceptibility.size)

# Configure benchmarks
# --------------------
# Create iterator
engines = ["choclo", "geoana"]
parallelism = [False, True]

iterators = (parallelism, engines)
pool = itertools.product(*iterators)

# Run benchmarks
# --------------
n_runs = 3

dims = ("engine", "parallel")
coords = {"engine": engines, "parallel": parallelism}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)

for index, (parallel, engine) in enumerate(pool):
    print(f"parallel: {parallel} engine: {engine}")
    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        chiMap=model_map,
        engine=engine,
        store_sensitivities="ram",
    )
    if engine == "choclo":
        kwargs["numba_parallel"] = parallel
    else:
        if parallel:
            kwargs["n_processes"] = None
        else:
            kwargs["n_processes"] = 1

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)
    runtime, std = benchmarker.benchmark(susceptibility)
    print(f"   {runtime} +/- {std} s")

    # Save results
    results.times.loc[engine, parallel] = runtime
    results.times_std.loc[engine, parallel] = runtime

print(results)
