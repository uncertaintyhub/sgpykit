import numpy as np

from sgpykit.util import matlab


def lev2knots_doubling(I):
    """
    Convert level index to number of knots using a doubling rule.

    This function maps a level index (or an array of indices) to the corresponding
    number of knots based on a doubling rule. The mapping is defined as:
        m = 2^{i-1} + 1 for i > 1
        m = 1 for i = 1
        m = 0 for i = 0

    Parameters
    ----------
    I : int or array_like
        Level(s) for which to compute the number of knots. Must be numeric.

    Returns
    -------
    m : int or ndarray
        Number of knots corresponding to the input level(s). If `I` is a scalar,
        returns a scalar. If `I` is an array, returns an array of the same shape.

    Notes
    -----
    The function handles both scalar and array inputs. For array inputs, the output
    array will have the same shape as the input array.
    """
    # level i>0 (i==0 will be sanitized), i can be a vector
    # m = lev2knots_doubling(i)
    # relation level / number of points:
    #    m = 2^{i-1}+1, for i>1
    #    m=1            for i=1
    #    m=0            for i=0
    # i.e. m(i)=2*m(i-1)-1
    assert matlab.isnumeric(I)
    i = np.atleast_1d(I)
    isarray = hasattr(I, '__iter__')
    m = np.zeros(i.shape, dtype='int64')
    m[i==1] = 1
    m[i>1] = 2**(i[i>1]-1)+1
    return m if isarray else m[0]
