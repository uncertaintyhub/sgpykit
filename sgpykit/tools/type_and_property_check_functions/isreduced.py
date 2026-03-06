from sgpykit.util import matlab


def isreduced(S):
    """
    Check if a given object is a reduced sparse grid.

    A reduced sparse grid is a struct with fields 'knots', 'm', 'weights', 'n', 'size'.

    Parameters
    ----------
    S : object
        The object to check.

    Returns
    -------
    int
        1 if S is a reduced sparse grid, 0 otherwise.
    """
    sparsegrid_fieldnames = {'knots', 'weights', 'size', 'n', 'm'}
    if matlab.isstruct(S) and \
            len(S) == 1 and sparsegrid_fieldnames.issubset(matlab.fieldnames(S)):
        return 1
    return 0

