import numpy as np


def knots_trap(n, x_a, x_b, whichrho='prob'):
    """
    Compute knots and weights for the trapezoidal quadrature rule.

    This function implements the univariate n-points trapezoidal quadrature rule
    with weight function rho(x) = 1/(x_b-x_a), i.e., x = linspace(x_a,x_b,n)
    and w = 1/(x_b - x_a) * [h/2 h h ... h h/2], with h = (x_b - x_a)/(n-1).

    Parameters
    ----------
    n : int
        Number of points in the quadrature rule.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.
    whichrho : str, optional
        Type of weight function. Options are:
        - 'prob': weights are normalized by the interval length (default).
        - 'nonprob': weights are not normalized.

    Returns
    -------
    x : numpy.ndarray
        Array of knots (quadrature points).
    w : numpy.ndarray
        Array of weights.

    Notes
    -----
    For n=1, the midpoint rule is used.
    """
    assert n > 0

    if n == 1:
        # Midpoint rule
        h = x_b - x_a
        x = (x_a + x_b) / 2
        w = h
    else:
        h = (x_b - x_a) / (n - 1)
        x = np.linspace(x_a, x_b, n)
        w = np.array([h / 2] + [h] * (n - 2) + [h / 2])

    if 'nonprob' == whichrho:
        pass
    elif 'prob' == whichrho:
        w = w / (x_b - x_a)
    else:
        raise ValueError('whichrho not recognized')

    return x, w
