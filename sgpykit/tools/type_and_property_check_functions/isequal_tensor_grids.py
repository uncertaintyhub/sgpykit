import numpy as np

from sgpykit.tools.type_and_property_check_functions.is_tensor_grid import is_tensor_grid


def isequal_tensor_grids(T1, T2, tol=1e-14):
    """
    Compare two tensor grids field by field, checking that the two grids are identical
    and accounting for numerical tolerance when comparing knots and weights.

    Parameters
    ----------
    T1 : tensor grid
        First tensor grid to compare.
    T2 : tensor grid
        Second tensor grid to compare.
    tol : float, optional
        Tolerance used when comparing points and weights. Default is 1e-14.

    Returns
    -------
    iseq : bool
        True if the tensor grids are equal, False otherwise.
    whatfield : str
        The name of the first field of the two tensor grids that are found different,
        or 'size' if the two sparse grids have a different number of tensor grids.
    """
    if is_tensor_grid(T1) == 0 or is_tensor_grid(T2) == 0:
        raise ValueError('T1 or T2 not tensor grids in isequal_tensor_grids')

    if not T1.size == T2.size:
        iseq = False
        whatfield = 'size'
        return iseq, whatfield

    if not np.all(T1.m == T2.m):
        iseq = False
        whatfield = 'm'
        return iseq, whatfield

    if np.any(np.abs(T1.weights - T2.weights) > tol):
        iseq = False
        whatfield = 'weights'
        return iseq, whatfield

    N = len(T1.knots_per_dim)
    N2 = len(T2.knots_per_dim)
    if N != N2:
        iseq = False
        whatfield = 'knots_per_dim'
        return iseq, whatfield

    for n in range(N):
        if np.any(np.abs(np.array(T1.knots_per_dim[n]) - np.array(T2.knots_per_dim[n])) > tol):
            iseq = False
            whatfield = 'knots_per_dim'
            return iseq, whatfield

    if np.any(np.abs(T1.knots - T2.knots) > tol):
        iseq = False
        whatfield = 'knots'
        return iseq, whatfield

    iseq = True
    whatfield = ''
    return iseq, whatfield
