import numpy as np

from sgpykit.tools.polynomials_functions.jacobi_prob_eval import jacobi_prob_eval
from sgpykit.util import matlab


def jacobi_prob_eval_multidim(X, k, alpha, beta, a, b):
    """
    Evaluate the multidimensional probabilistic Jacobi polynomial of order k
    (multi-index) orthonormal on [a1,b1] x [a2,b2] x [a3,b3] x ... [aN,bN]
    with respect to
    rho=prod_i Gamma(alpha_i+beta_i+2)/(Gamma(alpha_i+1)*Gamma(beta_i+1)*(b-a)^(alpha_i+beta_i+1))*(x-a)^alpha_i*(b-x)^beta_i,
    alpha_i,beta_i>-1 on the list of points X (each column is a point in R^N).

    Parameters
    ----------
    X : numpy.ndarray
        Array of points, shape (N, nb_pts), where N is the dimension and nb_pts is the number of points.
    k : numpy.ndarray
        Multi-index of the polynomial, shape (N,).
    alpha : numpy.ndarray or float
        Parameter alpha for the Jacobi polynomial, can be scalar or array of shape (N,).
    beta : numpy.ndarray or float
        Parameter beta for the Jacobi polynomial, can be scalar or array of shape (N,).
    a : numpy.ndarray or float
        Lower bound of the interval, can be scalar or array of shape (N,).
    b : numpy.ndarray or float
        Upper bound of the interval, can be scalar or array of shape (N,).

    Returns
    -------
    L : numpy.ndarray
        Evaluated polynomial values, shape (nb_pts,).
    """
    N, nb_pts = X.shape

    # take care of the fact that alpha,beta may be scalar values

    if len(alpha) == 1:
        alpha = np.full((1, N), alpha)
        beta = np.full((1, N), beta)

    if len(a) == 1:
        a = np.full((1, N), a)
        b = np.full((1, N), b)

    # L is a product of N polynomials, one for each dim. We store the evaluation
    # of each of these polynomials as rows of a matrix. We do not compute the
    # polynomials for k=0 though, we know already they will be 1

    nonzero_n = matlab.find(k != 0)
    if len(nonzero_n) == 0:
        L = np.ones(nb_pts)
    else:
        M = np.zeros((len(nonzero_n), nb_pts))
        for j,n in enumerate(nonzero_n):
            M[j, :] = jacobi_prob_eval(X[n,:], k[n], alpha[n], beta[n], a[n], b[n])
        L = np.prod(M, 0)

    return L
