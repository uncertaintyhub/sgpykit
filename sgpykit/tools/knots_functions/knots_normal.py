import numpy as np

from sgpykit.util import matlab


def knots_normal(n, mi, sigma):
    """
    Calculate the collocation points (x) and weights (w) for Gaussian integration
    with respect to the weight function rho(x) = 1/sqrt(2*pi*sigma^2) * exp(-(x-mi)^2 / (2*sigma^2)),
    i.e., the density of a Gaussian random variable with mean mi and standard deviation sigma.

    Parameters
    ----------
    n : int
        Number of collocation points.
    mi : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    x : ndarray
        Collocation points.
    w : ndarray
        Weights for Gaussian integration.
    """
    # [x,w]=KNOTS_NORMAL(n,mi,sigma)
    # calculates the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function
    # rho(x)=1/sqrt(2*pi*sigma^2) *exp( -(x-mi)^2 / (2*sigma^2) )
    # i.e. the density of a gaussian random variable
    # with mean mi and standard deviation sigma
    assert n>0
    if n == 1:
        # The point (translated if needed)
        x = mi
        # The weight is 1:
        w = 1
        return x, w

    # Calculates the values of the recursive relation
    a, b = coefherm(n)

    # Builds the matrix
    JacM = np.diag(a) + np.diag(np.sqrt(b[1:]), 1) + np.diag(np.sqrt(b[1:]), -1)

    # Calculates points and weights from eigenvalues / eigenvectors of JacM
    W, x = matlab.eig(JacM)
    w = W[0, :].real ** 2  # Extract the real part and square it

    # Sort the points and reorder the weights accordingly
    ind = np.argsort(x)
    x = x[ind]
    w = w[ind]

    # Modifies points according to mi, sigma (the weights are unaffected)
    x = mi + np.sqrt(2) * sigma * x

    return x, w


def coefherm(n):
    """
    Compute the recurrence coefficients for Hermite polynomials.

    Parameters
    ----------
    n : int
        Order of the Hermite polynomial.

    Returns
    -------
    a : ndarray
        Coefficients for the diagonal of the Jacobi matrix.
    b : ndarray
        Coefficients for the off-diagonal of the Jacobi matrix.
    """
    if n <= 1:
        raise ValueError('n must be > 1')

    a = np.zeros(n)
    b = np.zeros(n)

    b[0] = np.sqrt(np.pi)
    k = np.arange(2, n + 1)
    b[k - 1] = 0.5 * (k - 1)

    return a, b
