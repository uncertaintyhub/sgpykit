import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sgpykit.util.misc import merge_all_args_to_kwargs

logger_ = logging.getLogger(__name__)


def grid(flag='on'):
    if flag != 'on':
        raise NotImplementedError()
    # Turn on major gridlines with customization
    plt.grid(which='major', color='grey', linestyle='-', linewidth=1)

    # Optionally, turn on minor gridlines as well
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)


def set_legend(location):
    """
    Set the legend location in Matplotlib based on MATLAB's legend location strings.

    Parameters
    ----------
    location : str
        The desired location for the legend. Options include:
        'NorthEast', 'NorthWest', 'SouthEast', 'SouthWest',
        'North', 'South', 'East', 'West', 'Best',
        'NorthOutside', 'SouthOutside', 'EastOutside', 'WestOutside'.

    Raises
    ------
    ValueError
        If the provided location is not valid.
    """
    # Mapping MATLAB locations to Matplotlib locations
    location_map = {
        'NorthEast': ('upper right', (1.05, 1)),
        'NorthWest': ('upper left', (-0.05, 1)),
        'SouthEast': ('lower right', (1.05, 0)),
        'SouthWest': ('lower left', (-0.05, 0)),
        'North': ('upper center', (0.5, 1)),
        'South': ('lower center', (0.5, 0)),
        'East': ('center left', (1, 0.5)),
        'West': ('center right', (-0.05, 0.5)),
        'Best': ('best', (0.5, 0.5)),  # Best fits automatically
        'NorthOutside': ('upper center', (0.5, 1.05)),
        'SouthOutside': ('upper center', (0.5, -0.1)),
        'EastOutside': ('center left', (1.05, 0.5)),
        'WestOutside': ('center right', (-0.1, 0.5))
    }

    if location not in location_map:
        raise ValueError(f"Invalid location '{location}'. Valid options are: {list(location_map.keys())}")

    loc, bbox = location_map[location]
    plt.legend(loc=loc, bbox_to_anchor=bbox)


def view(azimuth, elevation):
    """
    Set the view angle for a 3D plot in Matplotlib.

    Parameters
    ----------
    azimuth : float
        The azimuthal angle in degrees.
    elevation : float
        The elevation angle in degrees.
    """
    plt.view_init(elev=elevation, azim=azimuth)


def clear():
    plt.clf()


def gcf():
    return plt.gcf()


def figure_create(nrows=1, ncols=1, figsize=(10,8), dims=2): # TODO: dims -> plot3d=False
    # returns fig, axs (nrows x ncols array)
    if dims == 2:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': '3d'})
    return fig, axs


def plot(ax, x, y, *args, **kwargs):
    """
    Plot data on a given axis with optional formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    x : array_like
        The x-coordinates of the data points.
    y : array_like
        The y-coordinates of the data points.
    *args : tuple
        Additional positional arguments for formatting.
    **kwargs : dict
        Additional keyword arguments for formatting.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The plotted line objects.
    """
    # Plotting based on the number of arguments
    if len(args) == 0 and len(kwargs) == 0:
        h = ax.plot(x, y)
    else:
        fmt, mplargs = parse_matlab_plot_args(*args, **kwargs)
        if fmt:
            h = ax.plot(x, y, fmt, **mplargs)
        else:
            # TODO: ok? <- default plot command should plot lines; user must specify other style explicitly
            # fmt = ''
            # if 'color' not in mplargs.keys():
            #     fmt += 'k'
            # if 'marker' not in mplargs.keys():
            #     fmt += 'o'
            # if fmt:
            #     h = ax.plot(x, y, fmt, linestyle='none', **mplargs)
            # else:
            h = ax.plot(x, y, **mplargs)
        #ax.set_title(axtitle)
    ax.grid(True)
    return h


def plot_scatter_3d(ax, x, y, z, *args, **kwargs):
    """
    Create a 3D scatter plot on a given axis with optional formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axis to plot on.
    x : array_like
        The x-coordinates of the data points.
    y : array_like
        The y-coordinates of the data points.
    z : array_like
        The z-coordinates of the data points.
    *args : tuple
        Additional positional arguments for formatting.
    **kwargs : dict
        Additional keyword arguments for formatting.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot object.
    """
    # Plotting based on the number of arguments
    if len(args) == 0 and len(kwargs) == 0:
        h = ax.scatter_3d(x, y, z, 'ok', markerfacecolor='k')
    else:
        fmt, mplargs = parse_matlab_plot_args(*args, **kwargs)
        if fmt:
            h = ax.scatter_3d(x, y, z, fmt, linestyle='none', **mplargs)
        else:
            h = ax.scatter_3d(x, y, z, linestyle='none', **mplargs)
    return h


def plot_3d(ax, x, y, z, *args, **kwargs):
    """
    Create a 3D line plot on a given axis with optional formatting.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        The 3D axis to plot on.
    x : array_like
        The x-coordinates of the data points.
    y : array_like
        The y-coordinates of the data points.
    z : array_like
        The z-coordinates of the data points.
    *args : tuple
        Additional positional arguments for formatting.
    **kwargs : dict
        Additional keyword arguments for formatting.

    Returns
    -------
    matplotlib.lines.Line2D
        The plotted line object.
    """
    # Plotting based on the number of arguments
    if len(args) == 0 and len(kwargs) == 0:
        h = ax.plot(x, y, z, 'ok', markerfacecolor='k')
    else:
        fmt, mplargs = parse_matlab_plot_args(*args, **kwargs)
        if fmt:
            h = ax.plot(x, y, z, fmt, **mplargs)
        else:
            h = ax.plot(x, y, z, **mplargs)
    return h


def parse_matlab_plot_args(*args, **kwargs):
    """
    Parse MATLAB-style plot arguments into a format string and keyword arguments.

    Parameters
    ----------
    *args : tuple
        Positional arguments for formatting.
    **kwargs : dict
        Keyword arguments for formatting.

    Returns
    -------
    tuple
        A tuple containing the format string and a dictionary of keyword arguments.
    """
    kwargs_all, fmt = merge_all_args_to_kwargs(args, kwargs, to_lowercase=True)
    # axtitle = kwargs.pop('???', 'Figure')
    if 'displayname' in kwargs_all:
        kwargs_all['label'] = kwargs_all.pop('displayname')
    return fmt, kwargs_all


def check_and_convert_to_fig3d(ax):
    """
    Check if the given axis is a 3D axis, and convert it if necessary.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to check and convert.

    Returns
    -------
    matplotlib.axes.Axes3D
        The 3D axis.
    """
    if isinstance(ax, Axes3D):
        return ax
    else:
        logger_.warning("Please directly specify sg.figure_create(dims=3) when using 3d plots.")
        fig = ax.figure
        # Create a new 3D Axes
        ax_3d = fig.add_subplot(ax.get_subplotspec(), projection='3d')
        ax_3d.set_title(ax.get_title())
        # TODO: more props to be copied?
        return ax_3d
