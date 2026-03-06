import numpy as np
from scipy.special import gamma

from sgpykit.util import matlab


def knots_gamma(n, alpha, beta):
    """
    Compute collocation points and weights for Gaussian integration w.r.t. the Gamma density.

    Returns the collocation points (x) and the weights (w) for the Gaussian integration
    with respect to the weight function:
    rho(x) = beta^(alpha+1)/Gamma(alpha+1) * x^alpha * exp(-beta*x)
    i.e., the density of a Gamma random variable with range [0,inf), alpha > -1, beta > 0.

    Parameters
    ----------
    n : int
        Number of collocation points.
    alpha : float
        Shape parameter of the Gamma distribution.
    beta : float
        Rate parameter of the Gamma distribution.

    Returns
    -------
    x : ndarray
        Collocation points.
    w : ndarray
        Weights for Gaussian integration.
    """
    assert n > 0
    if n == 1:
        # the point
        x = (alpha + 1) / beta
        # the weight is 1:
        w = 1
        return x, w

    # calculates the values of the recursive relation
    a, b = coeflagu_generalized(n, alpha)
    # builds the matrix
    JacM = np.diag(a) + np.diag(np.sqrt(b[1:n]), 1) + np.diag(np.sqrt(b[1:n]), -1)
    # calculates points and weights from eigenvalues / eigenvectors of JacM
    W, x = matlab.eig(JacM)

    w = W[0,:] ** 2
    x, ind = matlab.sort(x)

    w = w[ind]
    # rescales points
    x = x / beta
    # ----------------------------------------------------------------------
    return x,w


def coeflagu_generalized(n, alpha):
    """
    Compute recurrence coefficients for generalized Laguerre polynomials.

    Parameters
    ----------
    n : int
        Number of coefficients to compute.
    alpha : float
        Shape parameter of the Gamma distribution.

    Returns
    -------
    a : ndarray
        Diagonal coefficients.
    b : ndarray
        Off-diagonal coefficients.
    """
    if n <= 1:
        raise ValueError('n must be > 1')

    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = alpha + 1
    b[0] = gamma(1 + alpha)

    k = np.arange(1, n)
    a[1:] = 2 * k + alpha + 1
    b[1:] = k * (k + alpha)

    return a, b
