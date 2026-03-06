import numpy as np

from sgpykit.tools.polynomials_functions.herm_eval import herm_eval
from sgpykit.util import matlab


def herm_eval_multidim(X, k, mu, sig):
    """
    Evaluate the multidimensional Hermite polynomial of order k (multi-index)
    orthonormal on [-inf, +inf]^N with respect to the Gaussian weight function.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points where the polynomial is evaluated. Each column is a point in R^N.
    k : numpy.ndarray
        Multi-index defining the order of the polynomial in each dimension.
    mu : float or numpy.ndarray
        Mean of the Gaussian weight function. Can be a scalar or a vector of length N.
    sig : float or numpy.ndarray
        Standard deviation of the Gaussian weight function. Can be a scalar or a vector of length N.

    Returns
    -------
    H : numpy.ndarray
        Array of polynomial evaluations at the points in X.
    """
    # H = HERM_EVAL_MULTIDIM(X,k,mu,sig)
    # evaluates the multidim Hermite polynomial of order k (multi-index) orthonormal on [-inf,+inf]^N
    # with respect to rho=prod_i 1/sqrt(2 pi sigma_i^2) * e^( -(x-mi_i)^2/(2*sigma_i^2) )
    # on the list of points X (each column is a point in R^N)
    # MU, SIGMA, can be scalar values
    N, nb_pts = X.shape

    # take care of the fact that mu,sigma may be scalar values

    if len(mu) == 1:
        mu = np.full((1, N), mu)
        sig = np.full((1, N), sig)

    # H is a product of N polynomials, one for each dim. I store the evaluation
    # of each of these polynomials as rows of a matrix. I do not compute the
    # zeros though, I know already they will be 1

    nonzero_n = matlab.find(k != 0)
    if len(nonzero_n) == 0:
        H = np.ones((1, nb_pts))
    else:
        M = np.zeros((len(nonzero_n), nb_pts))
        j = 0
        for n in nonzero_n:
            M[j, :] = herm_eval(X[n,:], k[n], mu[n], sig[n])
            j = j + 1
        H = np.prod(M, 0)

    return H
