import gc
import numpy as np
import verde as vd
import discretize
from discretize.utils import mkvc
from simpeg.potential_fields import magnetics as simpeg_mag


def delete_simulation(simulation):
    """Properly delete a simulation"""
    del simulation._G
    del simulation
    gc.collect()


def get_region(mesh):
    """Get horizontal boundaries of the mesh."""
    xmin, xmax = mesh.nodes_x.min(), mesh.nodes_x.max()
    ymin, ymax = mesh.nodes_y.min(), mesh.nodes_y.max()
    return (xmin, xmax, ymin, ymax)


def create_tensor_mesh(shape, spacings):
    """Create a sample TensorMesh and a active_cells array for it."""
    # Create the TensorMesh
    h = [d * np.ones(s) for d, s in zip(spacings, shape)]
    origin = (0, 0, -shape[-1] * spacings[-1])
    mesh = discretize.TensorMesh(h, origin=origin)
    # Create active cells
    active_cells = np.ones(mesh.n_cells, dtype=bool)
    return mesh, active_cells


def create_susceptibilty(size, vector=False):
    """
    Create a random susceptibility array.

    Parameters
    ----------
    size : int
        Number of susceptibility values that will be generated.
        If ``vector`` is True, then the returning array will contain three
        times ``size`` elements.
    vector : bool, optional
        Whether to return effective susceptibility array (single array with
        3 times
        ``size`` elements), or a single susceptibility array (with ``size``
        elements)`.

    Returns
    -------
    numpy.ndarray
    """
    rng = np.random.default_rng(seed=42)
    if vector:
        size *= 3
    susceptibility = rng.uniform(low=1e-8, high=1e-2, size=size)
    return susceptibility


def create_observation_points(region, shape, height):
    """Create sample observation points."""
    grid_coords = vd.grid_coordinates(
        region=region, shape=shape, adjust="spacing", extra_coords=height
    )
    return grid_coords


def create_survey(grid_coords, components="tmi"):
    """Create a SimPEG magnetic survey with the observation points."""
    receiver_locations = np.array([mkvc(c.ravel().T) for c in grid_coords])
    receivers = simpeg_mag.receivers.Point(receiver_locations.T, components=components)
    source_field = simpeg_mag.UniformBackgroundField(
        receiver_list=[receivers], amplitude=55_000, declination=13, inclination=45
    )
    survey = simpeg_mag.survey.Survey(source_field)
    return survey
