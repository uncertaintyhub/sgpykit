import numpy as np

from sgpykit.util import matlab


def lev2knots_tripling(I):
    """
    Convert sparse-grid level index to number of knots using a tripling rule.

    This function maps a given level index (or array of indices) to the corresponding
    number of knots based on the tripling rule: m = 3^{i-1} for i > 0, and m = 0
    for i = 0.

    Parameters
    ----------
    I : int or array_like
        Level(s) for which to compute the number of knots. Must be non-negative. Must be numeric.

    Returns
    -------
    m : int or ndarray
        Number of knots corresponding to the input level(s). Returns a scalar if input
        is a scalar, otherwise returns a NumPy array of int64 values.

    Notes
    -----
    The tripling rule results in the following sequence of knot counts:
    - i = 0 -> m = 0
    - i = 1 -> m = 1
    - i = 2 -> m = 3
    - i = 3 -> m = 9
    - i = 4 -> m = 27
    - etc.
    """
    # m = lev2knots_tripling(i)
    # relation level / number of points:
    #    m = 3^{i-1}, for i>1
    #    m=0          for i=0
    # i.e. m = 1,3,9,27
    assert matlab.isnumeric(I)
    i = np.atleast_1d(I)
    isarray = hasattr(I, '__iter__')
    m = np.zeros(i.shape, dtype='int64')
    m[i>0] = 3**(i[i>0]-1)
    return m if isarray else m[0]
