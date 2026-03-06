import numpy as np


def knots_midpoint(n, x_a, x_b, whichrho='prob'):
    """
    Generate nodes and weights for the n-point midpoint quadrature rule.

    This function implements the univariate n-points midpoint quadrature rule,
    dividing the interval [x_a, x_b] into n subintervals of length h = (x_b - x_a)/n
    and returning the vector of midpoints as nodes. The weights are all equal to 1/n
    (probability weight function rho(x) = 1/(x_b-x_a)) or (x_b-x_a)/n (non-probability
    weight function rho(x) = 1).

    Parameters
    ----------
    n : int
        Number of points in the quadrature rule.
    x_a : float
        Left endpoint of the interval.
    x_b : float
        Right endpoint of the interval.
    whichrho : str, optional
        Type of weight function. Options are:
        - 'prob' (default): Probability weight function (weights sum to 1).
        - 'nonprob': Non-probability weight function (weights sum to (x_b - x_a)).

    Returns
    -------
    x : numpy.ndarray
        Array of nodes (midpoints of subintervals).
    w : numpy.ndarray
        Array of weights corresponding to the nodes.

    Raises
    ------
    ValueError
        If the 4th input (whichrho) is not recognized.
    """
    assert n > 0
    h = (x_b - x_a) / n
    x = np.linspace(x_a + h / 2, x_b - h / 2, n)

    if whichrho == 'nonprob':
        w = (x_b - x_a) / n * np.ones(n)
    elif whichrho == 'prob':
        w = 1 / n * np.ones(n)
    else:
        raise ValueError('4th input not recognized')

    return x, w
