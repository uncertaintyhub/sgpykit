from sgpykit.tools.knots_functions.constants_normal_leja import GaussianLejaPrecomputedKnotsAndWeights150, \
    Sym_GaussianLejaPrecomputedKnotsAndWeights150
from sgpykit.util import matlab


def knots_normal_leja(n, mi, sigma, type_):
    """
    Compute collocation points and weights for Gaussian-weighted Leja sequences.

    This function returns the collocation points (x) and the weights (w) for the
    weighted Leja sequence for integration with respect to the weight function
    rho(x) = 1/sqrt(2*pi*sigma^2) * exp(-(x-mi)^2 / (2*sigma^2)), i.e., the
    density of a Gaussian random variable with mean mi and standard deviation sigma.

    Knots and weights have been precomputed (up to 150) for the case mi=0 and
    sigma=1 following the work of A. Narayan and J. Jakeman, "Adaptive Leja sparse
    grid constructions for stochastic collocation and high-dimensional approximation",
    SIAM Journal on Scientific Computing, Vol. 36, No. 6, pp. A2952--A2983, 2014.

    If mi != 0 and sigma != 1, the precomputed knots are modified as follows:
    x = mi + sigma*x. The weights are unaffected.

    An error is raised if more than 150 points are requested. Knots are sorted
    increasingly before returning (weights are returned in the corresponding order).

    Parameters
    ----------
    n : int
        Number of collocation points.
    mi : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    type_ : str
        Type of Leja sequence. Supported types are:
        - 'line': Given X(1)=0, recursively defines the n-th point by
          X_n = argmax sqrt(rho(X)) * prod_{k=1}^{n-1} abs(X-X_k)
        - 'sym_line': Given X(1)=0, recursively defines the n-th and (n+1)-th
          point by X_n = argmax sqrt(rho(X)) * prod_{k=1}^{n-1} abs(X-X_k) and
          X_{n+1} = symmetric point of X_n with respect to 0.

    Returns
    -------
    x : numpy.ndarray
        Collocation points.
    w : numpy.ndarray
        Weights corresponding to the collocation points.

    Raises
    ------
    Exception
        If more than 150 points are requested.
    ValueError
        If an unknown Leja type is provided.
    """
    assert n > 0

    if n > 150:
        raise Exception(f'OutOfTable: this number of points is not available: {n}')
    elif 'line' == type_:
        X, W = GaussianLejaPrecomputedKnotsAndWeights150
        # --------------------------------------------------------
    elif 'sym_line' == type_:
        X, W = Sym_GaussianLejaPrecomputedKnotsAndWeights150
        # --------------------------------------------------------
    else:
        raise ValueError('unknown Leja type')

    x = X[:n]
    w = W[:n] # TODO: W[:n, n - 1] does not work on a 1d vector, also in matlab code

    # Modify points according to mi and sigma (the weights are unaffected)
    x = mi + sigma * x

    # Sort knots increasingly and weights accordingly. Weights need to be row vectors
    x, sorter = matlab.sort(x)
    w = w[sorter]

    return x, w
    
