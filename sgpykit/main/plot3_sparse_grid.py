import matplotlib.pyplot as plt
import numpy as np

from sgpykit.util.misc import reshape_nested_lists_to_nrows
from sgpykit.util.struct_array import StructArray


def plot3_sparse_grid(S, dims, *args, **kwargs):
    """
    Plot a sparse grid in 3D.

    Parameters
    ----------
    S : StructArray
        Sparse grid object to plot. S is a sparse grid in 3D. S can be either reduced or not. S can also be a tensor grid.
    dims : list
        List of dimensions to plot. It plots the components d1, d2, d3 of the points in S if S is more than 3D. If empty, defaults to [0, 1, 2].
    *args : tuple
        Additional positional arguments for plot styling.
    **kwargs : dict
        Additional keyword arguments for plot styling.

    Returns
    -------
    h : object
        Handle to the plot.
    """
    # Default dimensions if not provided
    if len(dims) == 0:
        dims = [0, 1, 2]

    # Extract knots from S
    x = S.knots
    if isinstance(S, StructArray):
        x = reshape_nested_lists_to_nrows(x, nrows=3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting based on the number of arguments
    if len(args) == 0 and len(kwargs) == 0:
        h = ax.scatter_3d(x[dims[0]], x[dims[1]], x[dims[2]])#, 'ok', markerfacecolor='k')
    else:
        if len(kwargs) == 0:
            kwargs = dict(np.array(args).reshape(-1, 2))
        # Convert all keys to lowercase strings
        # kwargs = {str(key).lower(): value for key, value in kwargs.items()}
        h = ax.scatter(x[dims[0]], x[dims[1]], x[dims[2]])

    #grid('on')  # Turn on grid
    return h
