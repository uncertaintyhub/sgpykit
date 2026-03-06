import numpy as np


def lagu_eval(x, k, lambda_):
    """
    Evaluate the k-th Laguerre polynomial orthonormal in [0, +inf) w.r.t. rho=lambda*exp(-lambda*x).

    This function computes the values of the k-th Laguerre polynomial, which is orthonormal with respect to the weight
    function rho = lambda * exp(-lambda * x) on the interval [0, +inf). The polynomials are defined
    starting from k=0, with L_0(x) = 1 and L_1(x) = -x*lambda + 1.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial (can be a matrix).
    k : int
        Degree of the Laguerre polynomial.
    lambda_ : float
        Parameter of the weight function (must be positive).

    Returns
    -------
    L : ndarray
        Values of the k-th Laguerre polynomial at the points x.
    """
    # L = lagu_eval(x,k,lambda)
    # returns the values of the k-th Laguerre polynomial orthoNORMAL in [0,+inf) w.r.t to rho=lambda*exp(-lambda*x), lambda>0,
    # in the points x (x can be a matrix as well)
    # N.B. the polynomials start from k=0: L_0(x) = 1, L_1(x) = - x*lambda + 1
    # this function expresses L as a function of the standard Laguerre "probabilistic" polynomial (i.e. orthoGONAL w.r.t. rho=e^(-x),
    # which are recursively calculated through the function standard_lagu_eval, coded below in this .m file

    # first compute the transformation of x (referred to Exp(lambda)) to z, the standard Exp(1)
    z = x * lambda_
    # calculate the standard Laguerre polynomials in z
    L = standard_lagu_eval(z, k)
    return L


def standard_lagu_eval(x, k):
    """
    Evaluate the k-th standard Laguerre "probabilistic" polynomial.

    This function computes the values of the k-th standard Laguerre polynomial, which is orthogonal with respect to the
    weight function rho = exp(-x) on the interval [0, +inf). The polynomials are defined starting from k=0,
    with L_0(x) = 1 and L_1(x) = -x + 1.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial (can be a vector).
    k : int
        Degree of the Laguerre polynomial (must be non-negative).

    Returns
    -------
    L : ndarray
        Values of the k-th standard Laguerre polynomial at the points x.
    """
    # L = standard_lagu_eval(x,k)
    # returns the values of the k-th standard Laguerre "probabilistic" polynomial (i.e. orthoGONAL w.r.t. rho=exp(-x), in the points x
    # ( x can be a vector as well)
    # N.B. the polynomials start from k=0: L_0(x) = 1, L_1(x) = -x+1
    assert k >= 0
    # base steps

    # read this as L(k-2)
    L_2 = np.ones_like(x)
    # and this as L(k-1)
    L_1 = -x + 1
    if k == 0:
        L = L_2
        return L
    elif k == 1:
        L = L_1
        return L
    else:
        L = None
        # recursive step ## TODO: optimize for vector ops?
        for ric in range(2, k+1):
            L = np.multiply((-x + 2 * (ric - 1) + 1) / ric, L_1) - (ric - 1) / ric * L_2
            L_2 = L_1
            L_1 = L
        return L
