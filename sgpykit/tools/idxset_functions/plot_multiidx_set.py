import numpy as np

from sgpykit.util.plot import plot, plot_3d

def plot_multiidx_set(ax, I, *args, **kwargs):
    """
    Plot the index set I for 2D and 3D cases.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    I : numpy.ndarray
        The index set to plot, shape (n, N) where N is the dimension.
    *args : tuple, optional
        Additional positional arguments passed to the plot function.
    **kwargs : dict, optional
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    h : matplotlib.collections.PathCollection or matplotlib.lines.Line2D
        The plot handle.

    Raises
    ------
    ValueError
        If the dimension of I is greater than 3.
    """
    # PLOT_MULTIIDX_SET(I) plots the index set I, in the case N=2 and N=3.
    # PLOT_MULTIIDX_SET(I,'PlotSpec',Spec_value, ...) specifies the plotting options to be used.

    N = I.shape[1]
    if 2 == N:
        if len(args) == 0 and len(kwargs) == 0:
            h = plot(ax, I[:, 0], I[:, 1], 'ok', 'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', 'k')
        else:
            h = plot(ax, I[:, 0], I[:, 1], *args, **kwargs)
        maxI = np.max(I)
        ax.set_xlim([0, maxI + 1])
        ax.set_ylim([0, maxI + 1])
        #plt.axis('square')
    elif 3 == N:
        if len(args) == 0 and len(kwargs) == 0:
            h = plot_3d(ax, I[:, 0], I[:, 1], I[:, 2], 'ok', 'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', 'k')
        else:
            h = plot_3d(ax, I[:, 0], I[:, 1], I[:, 2], *args, **kwargs)
        maxI = np.max(I)
        ax.set_xlim([0, maxI + 1])
        ax.set_ylim([0, maxI + 1])
        ax.set_zlim([0, maxI + 1])
    else:
        raise ValueError('cannot plot multidx with more than 3 dimensions')
    return h
