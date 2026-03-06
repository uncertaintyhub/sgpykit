import numpy as np

from sgpykit.tools.type_and_property_check_functions.is_sparse_grid import is_sparse_grid
from sgpykit.tools.type_and_property_check_functions.isequal_tensor_grids import isequal_tensor_grids


def isequal_sparse_grids(S1, S2, tol=1e-14):
    """
    Compare two sparse grids for equality.

    This function checks if two sparse grids are identical tensor grid by tensor grid,
    accounting for numerical tolerance when comparing knots and weights.

    Parameters
    ----------
    S1 : StructArray
        First sparse grid to compare.
    S2 : StructArray
        Second sparse grid to compare.
    tol : float, optional
        Tolerance used when comparing points and weights. Default is 1e-14.

    Returns
    -------
    iseq : bool
        True if the sparse grids are equal, False otherwise.
    whatfield : str
        Name of the first field found to be different, or 'length' if the sparse grids
        have a different number of tensor grids.

    Notes
    -----
    The function first checks if the inputs are valid sparse grids. Then, it compares the
    number of tensor grids in each sparse grid. If they differ, it returns False and 'length'.
    Otherwise, it iterates through each tensor grid, comparing them using `isequal_tensor_grids`.
    If any tensor grid is found to be different, it returns False and the name of the field
    that differs. If all tensor grids are equal, it returns True and an empty string.
    """
    if not is_sparse_grid(S1) or not is_sparse_grid(S2):
       raise ValueError('S1 or S2 not sparse grids in isequal_sparse_grids')

    iseq = False
    whatfield = ''
    L1 = len(S1)
    L2 = len(S2)
    if L1 != L2:
        iseq = False
        whatfield = 'length'
        return iseq, whatfield

    for l in np.arange(L1):
        iseq, whatfield = isequal_tensor_grids(S1[l], S2[l], tol)
        if not iseq:
            return iseq, whatfield
        if not np.all(S1[l].coeff == S2[l].coeff):
            iseq = False
            whatfield = 'coeff'
            return iseq, whatfield
        if not np.all(S1[l].idx == S2[l].idx):
            iseq = False
            whatfield = 'idx'
            return iseq, whatfield

    return iseq, whatfield
