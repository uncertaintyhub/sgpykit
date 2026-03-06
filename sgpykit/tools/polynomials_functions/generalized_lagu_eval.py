import numpy as np
from scipy.special import gamma, factorial


def generalized_lagu_eval(x, k, alpha, beta):
    """
    Evaluate the k-th generalized Laguerre polynomial orthonormal in [0, +inf).

    This function returns the values of the k-th generalized Laguerre polynomial
    orthonormal in [0, +inf) with respect to the weight function
    rho = beta^(alpha+1)/Gamma(alpha+1) * x^alpha * exp(-beta*x) at the points x.
    The polynomials start from k=0: L_0(x) = 1, L_1(x) = -x*beta + alpha + 1.
    This function expresses L as a function of the standard generalized Laguerre
    "probabilistic" polynomial (i.e., orthogonal w.r.t. rho = x^alpha * exp(-x)/Gamma(alpha+1)),
    which are recursively calculated through the function standard_generalized_lagu_eval.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial. Can be a matrix.
    k : int
        Degree of the polynomial.
    alpha : float
        Parameter of the generalized Laguerre polynomial.
    beta : float
        Parameter of the generalized Laguerre polynomial.

    Returns
    -------
    L : ndarray
        Values of the k-th generalized Laguerre polynomial at the points x.
    """
    # first compute the transformation of x (referred to Gamma(alpha,beta)) to z, the standard Gamma(alpha,1)
    z = x * beta
    # calculate the standard generalized Laguerre polynomials in z
    L = standard_generalized_lagu_eval(z, k, alpha)
    # modify L to take into account normalizations.
    if k >= 1:
        L = L / np.sqrt(gamma(k + alpha + 1) / (gamma(alpha + 1) * factorial(k)))

    return L


def standard_generalized_lagu_eval(x, k, alpha):
    """
    Evaluate the k-th standard generalized Laguerre "probabilistic" polynomial.

    This function returns the values of the k-th standard generalized Laguerre
    "probabilistic" polynomial (i.e., orthogonal w.r.t. rho = x^alpha * exp(-x)/Gamma(alpha+1))
    at the points x. The polynomials start from k=0: L_0(x) = 1, L_1(x) = -x + alpha + 1.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial. Can be a vector.
    k : int
        Degree of the polynomial.
    alpha : float
        Parameter of the generalized Laguerre polynomial.

    Returns
    -------
    L : ndarray
        Values of the k-th standard generalized Laguerre polynomial at the points x.
    """
    assert k >= 0
    # base steps

    # read this as L(k-2)
    L_2 = np.ones_like(x)
    # and this as L(k-1)
    L_1 = -x + alpha + 1
    if k == 0:
        L = L_2
        return L
    elif k == 1:
        L = L_1
        return L
    else:
        L = None
        # recursive step
        for ric in range(2, k + 1):
            L = np.multiply((-x + 2 * (ric - 1) + alpha + 1) / ric, L_1) - (ric - 1 + alpha) / ric * L_2
            L_2 = L_1
            L_1 = L
        return L
