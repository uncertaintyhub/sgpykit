import numpy as np

from sgpykit.util import matlab


def lev2knots_2step(I):
    """
    Convert level index to number of knots using the 2-step rule.

    The 2-step rule defines the number of knots as m = 2(i-1)+1 for i > 0,
    resulting in an odd number of points (1, 3, 5, ...).

    Parameters
    ----------
    I : int or array_like
        Level(s) for which to compute the number of knots. Must be numeric.

    Returns
    -------
    m : int or ndarray
        Number of points to be used in each direction. If input is scalar,
        returns scalar. If input is array, returns array of same shape.

    Notes
    -----
    - For i <= 0, the output is 0.
    - The function preserves the input type (scalar or array).
    """
    #   relation level / number of points:
    #    m = 2(i-1)+1,
    #   i.e. m=1,3,5,7,9,...
    #   [m] = lev2knots_2step(i)
    #   i: level in each direction
    #   m: number of points to be used in each direction
    assert matlab.isnumeric(I)
    i = np.atleast_1d(I)
    isarray = hasattr(I, '__iter__')
    m = np.zeros(i.shape, dtype='int64')
    m[i>0] = 2*(i[i>0]-1)+1
    return m if isarray else m[0]
