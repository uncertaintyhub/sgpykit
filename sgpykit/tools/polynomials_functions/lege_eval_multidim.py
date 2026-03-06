import numpy as np

from sgpykit.tools.polynomials_functions.lege_eval import lege_eval


def lege_eval_multidim(X, k, a, b):
    """
    Evaluate the multidimensional Legendre polynomial of order k (multi-index)
    orthonormal on [a,b]^N on the list of points X (each column is a point in R^N).
    a,b are scalar values.

    Parameters
    ----------
    X : numpy.ndarray
        Array of points, shape (N, nb_pts), where N is the dimension and nb_pts is the number of points.
    k : numpy.ndarray
        Multi-index of the polynomial, shape (N,).
    a : int or numpy.ndarray
        Lower bound(s) of the domain. If scalar, it is broadcasted to all dimensions.
    b : int or numpy.ndarray
        Upper bound(s) of the domain. If scalar, it is broadcasted to all dimensions.

    Returns
    -------
    L : numpy.ndarray
        Evaluated polynomial values, shape (nb_pts,).

    Notes
    -----
    a,b may differ in each direction, so that actually the domain is not [a,b]^N, but
    [a1,b1] x [a2,b2] x [a3,b3] x ... [aN,bN]
    in this case, a,b are defined as vectors, each with N components.
    """
    N, nb_pts = X.shape

    # L is a product of N polynomials, one for each dim. I store the evaluation
    # of each of these polynomials as columns of a matrix
    M = 0 * X
    # take care of the fact that a,b may be scalar values
    if isinstance(a, int):
        a = a * np.ones(N)
        b = b * np.ones(N)

    for n in range(N):
        M[n, :] = lege_eval(X[n,:], k[n], a[n], b[n])

    L = np.prod(M, 0)
    return L
