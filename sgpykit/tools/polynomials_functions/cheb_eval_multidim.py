import numpy as np

from sgpykit.tools.polynomials_functions.cheb_eval import cheb_eval


def cheb_eval_multidim(X, k, a, b):
    """
    Evaluate the multidimensional Chebyshev polynomial of the first kind of order k (multi-index) on [a,b]^N.

    Evaluates the multidim Chebyshev polynomial of the first kind of order k (multi-index) on [a,b]^N
    on the list of points X (each column is a point in R^N). a,b are scalar values
    a,b may differ in each direction, so that actually the domain is not [a,b]^N, but
    [a1,b1] x [a2,b2] x [a3,b3] x ... [aN,bN]
    in this case, a,b are defined as vectors, each with N components

    Parameters
    ----------
    X : numpy.ndarray
        Array of points, shape (N, nb_pts), where each column is a point in R^N.
    k : numpy.ndarray
        Multi-index of the polynomial order, shape (N,).
    a : numpy.ndarray or float
        Lower bounds of the domain. If scalar, it is expanded to a vector of length N.
    b : numpy.ndarray or float
        Upper bounds of the domain. If scalar, it is expanded to a vector of length N.

    Returns
    -------
    C : numpy.ndarray
        Evaluated polynomial values at each point, shape (nb_pts,).
    """
    N, nb_pts = X.shape

    # L is a product of N polynomials, one for each dim. I store the evaluation
    # of each of these polynomials as columns of a matrix

    M = 0 * X
    # take care of the fact that a,b may be scalar values

    if len(a) == 1:
        a = np.full((1, N), a)
        b = np.full((1, N), b)

    for n in range(N):
        M[n, :] = cheb_eval(X[n,:], k[n], a[n], b[n])
    C = np.prod(M, 0)
    return C
