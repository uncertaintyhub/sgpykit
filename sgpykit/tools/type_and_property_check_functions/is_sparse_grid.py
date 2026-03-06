from sgpykit.util import matlab


def is_sparse_grid(S):
    """
    Check if the input is a sparse grid.

    A sparse grid is a vector of structs with fields 'knots', 'weights', 'size',
    'knots_per_dim', 'm', 'coeff', 'idx'.

    Parameters
    ----------
    S : object
        Input object to check.

    Returns
    -------
    bool
        True if S is a sparse grid, False otherwise.
    """
    sparsegrid_fieldnames = {'knots', 'weights', 'size', 'knots_per_dim', 'm', 'coeff', 'idx'}
    if matlab.isstruct(S) and \
            len(S) >= 1 and sparsegrid_fieldnames.issubset(matlab.fieldnames(S)):
        return True
    return False
