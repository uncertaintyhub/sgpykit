import numpy as np


def lege_eval(x, k, a, b):
    """
    Evaluate the k-th orthonormal Legendre polynomial on the interval [a, b].

    This function computes the values of the k-th Legendre polynomial, which is
    orthonormal with respect to the uniform weight function rho = 1/(b-a), at the
    points specified by x. The polynomials are defined starting from k=0:
    L_0(x) = 1, L_1(x) = (2*x - a - b)/(b - a).

    The computation is performed by transforming the input points from the interval
    [a, b] to the standard interval [-1, 1], evaluating the standard Legendre
    polynomial (orthogonal with respect to rho=1) at these transformed points,
    and then applying the necessary normalizations to account for the interval [a, b].

    Parameters
    ----------
    x : array_like
        The points at which to evaluate the polynomial. Can be a scalar, list, or ndarray.
    k : int
        The degree of the Legendre polynomial.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.

    Returns
    -------
    L : ndarray
        The values of the k-th orthonormal Legendre polynomial evaluated at x.
    """
    # first compute the transformation of x in (a,b) to t in (-1,1)
    x_arr = np.asarray(x, dtype=float)  # allow scalars, lists, nd-arrays, etc.
    t = (2 * x_arr - a - b) / (b - a)

    # calculate the standard Legendre polynomials in t
    L = standard_lege_eval(t, k)

    # modify L to take into account normalizations
    st_lege_norm = np.sqrt(2.0 / (2 * k + 1))

    # moreover, add an additional sqrt(2) to take into account a general interval (a,b) , not (-1,1)
    L = np.sqrt(2.0) * L / st_lege_norm

    return L


def standard_lege_eval(x, k):
    """
    Evaluate the k-th standard Legendre polynomial.

    This function computes the values of the k-th standard Legendre polynomial,
    which is orthogonal (but not orthonormal) with respect to the uniform weight
    function rho=1 on the interval [-1, 1], at the points specified by x.
    The polynomials are defined starting from k=0: L_0(x) = 1, L_1(x) = x.

    Parameters
    ----------
    x : array_like
        The points at which to evaluate the polynomial. Can be a scalar, list, or ndarray.
    k : int
        The degree of the Legendre polynomial.

    Returns
    -------
    L : ndarray
        The values of the k-th standard Legendre polynomial evaluated at x.
    """
    assert k >= 0
    # base steps

    # read this as L(k-2)
    x_arr = np.asarray(x, dtype=float)
    L_2 = np.ones_like(x_arr)  # L_{k-2}
    # and this as L(k-1)
    L_1 = x_arr  # L_{k-1}

    if k == 0:
        L = L_2
        return L
    elif k == 1:
        L = L_1
        return L
    else:
        L = None
        # recursive step
        for ric in range(2, k + 1):
            L = ((2 * ric - 1) / ric) * x_arr * L_1 - ((ric - 1) / ric) * L_2
            L_2 = L_1
            L_1 = L
        return L
