"""
Functions to create xarray object for storing results
"""

import numpy as np
import xarray as xr


def create_dataset(dims, coords, data_names):
    """
    Create empty dataset for the passed dims and coords

    Parameters
    ----------
    dims : tuple of str
        Tuple of strings with the dimensions of the data arrays.
    coords : dict
        Dictionary of coordinates. Keys must be the strings in dims, values
        must be array_like objects.
    data_names : list of str
        List of names for the new data arrays in the dataset. They will all
        share the same dimensions.

    Returns
    -------
    xr.Dataset
    """
    dims_set, coords_set = set(dims), set(coords.keys())
    if dims_set != coords_set:
        raise ValueError(
            "Invalid 'dims' and 'coords'. "
            "Dimensions don't match keys in coordinates."
        )
    data_shape = tuple(len(coords[f]) for f in dims)
    data = {name: (dims, np.full(data_shape, np.nan)) for name in data_names}
    ds = xr.Dataset(data, coords=coords)
    return ds
