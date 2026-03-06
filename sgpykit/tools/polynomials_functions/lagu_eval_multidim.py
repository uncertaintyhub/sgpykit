import numpy as np

from sgpykit.tools.polynomials_functions.lagu_eval import lagu_eval
from sgpykit.util import matlab


def lagu_eval_multidim(X, k, lambda_):
    """
    Evaluate the multidimensional Laguerre polynomial of order k (multi-index)
    orthonormal on [0, +inf)^N with respect to rho=prod_i lambda_i*exp(-lambda_i*x),
    lambda_i>0 on the list of points X (each column is a point in R^N).
    Lambda can be a scalar value.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points, shape (N, nb_pts), where N is the dimension and nb_pts is the number of points.
    k : numpy.ndarray
        Multi-index of the polynomial, shape (N,).
    lambda_ : float or numpy.ndarray
        Scale parameter(s) for the Laguerre polynomial. If scalar, it is expanded to a vector of length N.

    Returns
    -------
    L : numpy.ndarray
        Evaluated polynomial values, shape (nb_pts,).
    """
    N, nb_pts = X.shape

    # take care of the fact that lambda may be a scalar value
    if len(lambda_) == 1:
        lambda_ = np.full((1, N), lambda_)

    # L is a product of N polynomials, one for each dim. We store the evaluation
    # of each of these polynomials as rows of a matrix. We do not compute the
    # polynomials for k=0 though, we know already they will be 1
    nonzero_n = matlab.find(k != 0)
    if len(nonzero_n) == 0:
        L = np.ones(nb_pts)
    else:
        M = np.zeros((len(nonzero_n), nb_pts))
        for j, n in enumerate(nonzero_n):
            M[j, :] = lagu_eval(X[n, :], k[n], lambda_[n])
        L = np.prod(M, 0)

    return L
