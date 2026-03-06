from sgpykit.tools.knots_functions.constants import TriangularLejaPrecomputedKnotsAndWeights50
from sgpykit.util import matlab


def knots_triangular_leja(n, a, b):
    """
    Returns the collocation points (x) and the weights (w) for the weighted Leja
    sequence for integration with respect to the weight function rho(x) = 2/(b-a)^2 (b-x),
    i.e., a linear decreasing pdf over the interval [a, b].

    Knots and weights have been precomputed (up to 50) following the work of
    A. Narayan, J. Jakeman, "Adaptive Leja sparse grid constructions for stochastic collocation
    and high-dimensional approximation", SIAM Journal on Scientific Computing, Vol. 36, No. 6,
    pp. A2952--A2983, 2014.

    Parameters
    ----------
    n : int
        Number of collocation points.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.

    Returns
    -------
    x : numpy.ndarray
        Collocation points.
    w : numpy.ndarray
        Weights corresponding to the collocation points.

    Raises
    ------
    Exception
        If more than 50 points are requested.
    """
    # An error is raised if more than 50 points are requested.
    assert n > 0
    if n > 50:
        raise Exception(f'OutOfTable: this number of points is not available: {n}')
    else:
        X, W = TriangularLejaPrecomputedKnotsAndWeights50
        x = X[:n]
        w = W[:n]
        # modifies points according to the interval (the weights are unaffected)
        x = (b - a) * x + a
        # sort knots increasingly and weights accordingly. Weights need to be row vectors
        x, sorter = matlab.sort(x)
        w = w[sorter]

    return x, w
