from sgpykit.util.misc import reshape_nested_lists_to_nrows
from sgpykit.util.plot import plot
from sgpykit.util.struct_array import StructArray


def plot_sparse_grid(ax, S, dims, *args, **kwargs):
    """
    Plot object based on the provided structure S and dimensions.

    Plot a sparse grid in 2D or higher dimensions by selecting the specified
    dimensions. The sparse grid can be either reduced or not, or even a tensor
    grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    S : StructArray
        An object containing 'knots' as an attribute. Typically a sparse grid
        structure.
    dims : list or None
        Dimensions to plot (default is [1, 2]). If S is more than 2D, this
        specifies which dimensions to plot.
    *args : tuple
        Additional positional arguments for plotting, such as format strings.
    **kwargs : dict
        Additional keyword arguments for plotting, such as 'color', 'marker', etc.

    Returns
    -------
    h : matplotlib.lines.Line2D
        The handle to the plotted object.

    """

    # Default dimensions if not provided
    if len(dims) == 0:
        dims = [1, 2]

    # Extract knots from S
    x = S.knots
    if isinstance(S, StructArray):
        x = reshape_nested_lists_to_nrows(x, nrows=2)

    if len(args) == 0 and len(kwargs) == 0:
        h = plot(ax, x[dims[0] - 1], x[dims[1] - 1], 'ok', markerfacecolor='k', linestyle='none')
    else:
        # the optional fmt string (first in args) will also be placed first when given to matplotlib
        h = plot(ax, x[dims[0] - 1], x[dims[1] - 1], linestyle='none', *args, **kwargs)
    return h
