from sgpykit.tools.knots_functions.constants import ExponentialLejaPrecomputedKnotsAndWeights50
from sgpykit.util import matlab


def knots_exponential_leja(n, lambda_):
    """
    Returns the collocation points (x) and the weights (w) for the weighted
    Leja sequence for integration with respect to the weight function
    rho(x) = exp(-lambda * abs(x)), lambda > 0, i.e., the density of an
    exponential random variable with rate parameter lambda.

    Knots and weights have been precomputed (up to 50) for the case lambda=1
    following the work:
    A. Narayan, J. Jakeman, "Adaptive Leja sparse grid constructions for
    stochastic collocation and high-dimensional approximation", SIAM Journal on
    Scientific Computing, Vol. 36, No. 6, pp. A2952--A2983, 2014.

    An error is raised if more than 50 points are requested. Knots are sorted
    increasingly before returning (weights are returned in the corresponding
    order).

    Parameters
    ----------
    n : int
        Number of collocation points.
    lambda_ : float
        Rate parameter of the exponential distribution.

    Returns
    -------
    x : numpy.ndarray
        Collocation points.
    w : numpy.ndarray
        Weights corresponding to the collocation points.

    Raises
    ------
    ValueError
        If more than 50 points are requested.
    """
    assert n > 0
    if n > 50:
        raise ValueError(f'OutOfTable: this number of points is not available: {n}')
    else:
        X, W = ExponentialLejaPrecomputedKnotsAndWeights50
        x = X[:n]
        w = W[:n]  # TODO: W[:n, n - 1] does not work on a 1d vector, also in matlab code)
        # modifies points according to lambda (the weigths are unaffected)
        x = x / lambda_
        # sort knots increasingly and weights accordingly. Weights need to be row vectors
        x, sorter = matlab.sort(x)
        w = w[sorter]

    return x, w
