import numpy as np

from sgpykit.tools.polynomials_functions.generalized_lagu_eval import generalized_lagu_eval
from sgpykit.util import matlab


def generalized_lagu_eval_multidim(X, k, alpha, beta):
    """
    Evaluate the multidimensional generalized Laguerre polynomial of order k (multi-index)
    orthonormal on [0, +inf)^N with respect to the weight function:
    rho = prod_i beta_i^(alpha_i+1)/Gamma(alpha_i+1) * x^alpha_i * exp(-beta_i * x),
    where alpha_i > -1 and beta_i > 0.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points where the polynomial is evaluated. Each column is a point in R^N.
    k : numpy.ndarray
        Multi-index defining the polynomial order.
    alpha : numpy.ndarray or scalar
        Parameter alpha for the generalized Laguerre polynomial. Can be scalar or array.
    beta : numpy.ndarray or scalar
        Parameter beta for the generalized Laguerre polynomial. Can be scalar or array.

    Returns
    -------
    L : numpy.ndarray
        Array of polynomial evaluations at the given points.
    """
    N, nb_pts = X.shape

    # Handle scalar alpha and beta by expanding them to arrays
    if len(alpha) == 1:
        alpha = np.full((1, N), alpha)
        beta = np.full((1, N), beta)

    # L is a product of N polynomials, one for each dimension.
    # We store the evaluation of each of these polynomials as rows of a matrix.
    # We do not compute the polynomials for k=0 though, as they will be 1.

    nonzero_n = matlab.find(k != 0)
    if len(nonzero_n) == 0:
        L = np.ones((1, nb_pts))
    else:
        M = np.zeros((len(nonzero_n), nb_pts))
        j = 0
        for n in nonzero_n:
            M[j, :] = generalized_lagu_eval(X[n,:], k[n], alpha[n], beta[n])
            j = j + 1
        L = np.prod(M, 0)

    return L
