import numpy as np

def export_sparse_grid_to_file(Sr, filename='points.dat', with_weights=False):
    """
    Save knots of a reduced sparse grid to an ASCII file.

    The first line of the file shows the number of points in the grid and their dimension.
    Then, points are stored as lines. If with_weights is True, the corresponding weight
    is added as the last entry of the row.

    Parameters
    ----------
    Sr : object
        Reduced sparse grid object containing knots and weights.
    filename : str, optional
        Name of the file to save the points to. Default is 'points.dat'.
    with_weights : bool, optional
        If True, weights are saved alongside the points. Default is False.

    Notes
    -----
    - If with_weights is False, the file format is:
        num_points dimension
        coord1(P1) coord2(P1) ... coordN(P1)
        coord1(P2) coord2(P2) ... coordN(P2)
        ...
    - If with_weights is True, the file format is:
        num_points dimension
        coord1(P1) coord2(P1) ... coordN(P1) weight(P1)
        coord1(P2) coord2(P2) ... coordN(P2) weight(P2)
        ...
    """
    # Combine header and data in memory before writing
    header = f"{Sr.knots.shape[1]} {Sr.knots.shape[0]}\n" # ok since we gonna transpose

    if with_weights:
        data = np.vstack((Sr.knots, Sr.weights)).T
    else:
        data = Sr.knots.T

    # Save it
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt='%.16e', delimiter=' ')
