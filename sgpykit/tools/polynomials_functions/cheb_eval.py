import numpy as np


def cheb_eval(x, k, a, b):
    """
    Evaluate the k-th Chebyshev polynomial of the first kind on [a,b].

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the Chebyshev polynomial.
    k : int
        Degree of the Chebyshev polynomial.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.

    Returns
    -------
    L : ndarray
        Values of the k-th Chebyshev polynomial evaluated at x.
    """
    # L = cheb_eval(x,k,a,b)
    # returns the values of the k-th Chebyshev polynomial of the first kind
    # on [a,b] (i.e. T_n(a)=(-1)^n, T_n(b)=1), evaluated at x (vector)
    assert k >= 0
    # first compute the transformation of x in (a,b) to t in (-1,1)
    t = (2 * x - a - b) / (b - a)
    # base steps
    # read this as L(k-2)
    L_2 = np.ones_like(t)
    # and this as L(k-1)
    L_1 = t
    if k == 0:
        L = L_2
        return L
    elif k == 1:
        L = L_1
        return L
    else:
        L = None
        # recursive step
        for _ in range(1,k):
            L = 2 * t * L_1 - L_2
            L_2 = L_1
            L_1 = L
        return L
