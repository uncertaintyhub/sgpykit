import matplotlib.pyplot as plt


def plot_idx_status(G, I, idx_bin, idx):
    """
    Plot the status of a two-dimensional adaptive sparse grid.

    This function visualizes the indices used in the construction of a two-dimensional
    adaptive sparse grid. It plots four sets of points:
    - G: set of indices used to build the sparse grid
    - I: set of indices whose neighbors have been explored
    - idx_bin: set of indices whose neighbors have not been explored
    - idx: next index to be considered

    Parameters
    ----------
    G : numpy.ndarray
        Array of shape (n, 2) containing the indices used to build the sparse grid.
    I : numpy.ndarray
        Array of shape (m, 2) containing the indices whose neighbors have been explored.
    idx_bin : numpy.ndarray
        Array of shape (p, 2) containing the indices whose neighbors have not been explored.
    idx : numpy.ndarray
        Array of shape (2,) containing the next index to be considered.

    Returns
    -------
    None
    """
    # plots status of a two-dimensional adaptive sparse grid
    # plot the indices used
    fig, ax = plt.subplots()
    ax.plot(G[:, 0], G[:, 1], 'xr', linewidth=2, markersize=14, label='G -- set used to build the sparse grid ')
    ax.plot(I[:, 0], I[:, 1], 'o', markerfacecolor='b', markersize=8, label='I -- set of idx whose neighbour \n has been explored ')
    if idx_bin.size > 0:
        ax.plot(idx_bin[:, 0], idx_bin[:, 1], 'sk', markerfacecolor='none', markersize=14, label='idx-bin -- set of idx whose \n neighbour has not been explored ')
    ax.plot(idx[0], idx[1], 'og', markerfacecolor='g', markersize=8, label='next idx to be considered')
    ax.tick_params(labelsize=16)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
