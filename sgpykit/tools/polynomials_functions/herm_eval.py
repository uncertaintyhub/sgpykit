import numpy as np
from scipy.special import factorial


def herm_eval(x, k, mi, sigma):
    """
    Evaluate the k-th orthonormal Hermite polynomial at points x.

    This function returns the values of the k-th Hermite polynomial orthonormal in
    (-inf, +inf) with respect to the weight function rho = 1/sqrt(2 pi sigma^2) *
    exp(-(x-mi)^2/(2*sigma^2)) at the points x (x can be a matrix as well).

    The polynomials start from k=0: H_0(x) = 1, H_1(x) = (x - mi)/sigma.
    This function expresses H as a function of the standard Hermite "probabilistic"
    polynomial (i.e., orthogonal with respect to rho=1/sqrt(2 pi) * exp(-x^2/2)),
    which are recursively calculated through the function standard_herm_eval.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial.
    k : int
        Degree of the Hermite polynomial.
    mi : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    H : ndarray
        Values of the k-th orthonormal Hermite polynomial at points x.
    """
    # first compute the transformation of x (referred to N(mi,sigma^2)) to z, the standard gaussian
    z = (x - mi) / sigma
    # calculate the standard legendre polynomials in t
    H = standard_herm_eval(z, k)
    # modify H to take into account normalizations.
    if k > 1:
        H = H / np.sqrt(factorial(k))

    return H


def standard_herm_eval(x, k):
    """
    Evaluate the k-th standard Hermite "probabilistic" polynomial at points x.

    This function returns the values of the k-th standard Hermite "probabilistic"
    polynomial (i.e., orthogonal with respect to rho=1/sqrt(2 pi) * exp(-x^2/2))
    at the points x (x can be a vector as well).

    The polynomials start from k=0: H_0(x) = 1, H_1(x) = x.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial.
    k : int
        Degree of the Hermite polynomial.

    Returns
    -------
    H : ndarray
        Values of the k-th standard Hermite polynomial at points x.
    """
    assert k >= 0
    # base steps

    # read this as H(k-2)
    H_2 = np.ones_like(x)
    # and this as H(k-1)
    H_1 = x
    if k == 0:
        H = H_2
        return H
    elif k == 1:
        H = H_1
        return H
    else:
        H = None
        # recursive step
        for ric in range(2, k+1):
            H = x * H_1 - (ric - 1) * H_2
            H_2 = H_1
            H_1 = H
        return H
