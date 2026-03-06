import numpy as np
from scipy.special import gamma, factorial


def jacobi_prob_eval(x, k, alpha, beta, a, b):
    """
    Evaluate the k-th Jacobi probabilistic polynomial orthonormal in [a,b].

    This function returns the values of the k-th Jacobi "probabilistic" polynomial
    orthonormal in [a,b] with respect to the weight function:
    rho(x) = Gamma(alpha+beta+2)/(Gamma(alpha+1)*Gamma(beta+1)*(b-a)^(alpha+beta+1)) * (x-a)^alpha * (b-x)^beta
    in the points x (x can be a matrix as well).

    Note that these are NOT the classical Jacobi polynomials (which are orthonormal
    in [a,b] w.r.t. (b-x)^alpha*(x-a)^beta).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial.
    k : int
        Degree of the polynomial.
    alpha : float
        Parameter of the Jacobi polynomial.
    beta : float
        Parameter of the Jacobi polynomial.
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.

    Returns
    -------
    L : ndarray
        Values of the k-th Jacobi probabilistic polynomial at points x.

    Notes
    -----
    The polynomials start from k=0: L_0(x) = 1,
    L_1(x) = 0.5*(alpha-beta) + 0.5*(alpha+beta+2)*(-2*x+a+b)/(b-a)

    This function expresses L as a function of the standard Jacobi polynomial (i.e. orthoGONAL w.r.t.
    rho(x)=(1-x)^alpha*(1+x)^beta, which are recursively calculated through the function standard_jac_eval.
    """
    z = (- 2 * x + a + b) / (b - a)
    # calculate the standard Jacobi polynomials in z
    L = standard_jac_eval(z, k, alpha, beta)
    # modify L to take into account normalizations.
    if k >= 1:
        C = gamma(alpha + beta + 2) * gamma(k + alpha + 1) * gamma(
            k + beta + 1) / ((2 * k + alpha + beta + 1) * gamma(k + alpha + beta + 1) * factorial(
            k) * gamma(alpha + 1) * gamma(beta + 1))
        L = L / np.sqrt(C)
    return L

def standard_jac_eval(x, k, alpha, beta):
    """
    Evaluate the k-th standard Jacobi polynomial orthogonal in [-1,1].

    This function returns the values of the k-th standard Jacobi polynomial (i.e. orthoGONAL in [-1,1] w.r.t.
    rho(x)=(1-x)^alpha*(1+x)^beta in the points x (x can be a vector as well).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the polynomial.
    k : int
        Degree of the polynomial.
    alpha : float
        Parameter of the Jacobi polynomial.
    beta : float
        Parameter of the Jacobi polynomial.

    Returns
    -------
    L : ndarray
        Values of the k-th standard Jacobi polynomial at points x.

    Notes
    -----
    The polynomials start from k=0: L_0(x) = 1, L_1(x) = 0.5*(alpha-beta) + 0.5*(alpha+beta+2)*x
    """
    assert k >= 0
    # base steps

    # read this as L(k-2)
    L_2 = np.ones_like(x)
    # and this as L(k-1)
    L_1 = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2) * x
    if k == 0:
        L = L_2
        return L
    else:
        if k == 1:
            L = L_1
            return L
        else:
            L = None
            # recursive step
            for ric in range(2, k+1):
                c1 = np.multiply(np.multiply(2 * ric, (2 * (ric - 1) + alpha + beta)), (ric - 1 + alpha + beta + 1))
                c2 = 2 * (ric - 1) + alpha + beta + 1
                c3 = (2 * (ric - 1) + alpha + beta + 2) * (2 * (ric - 1) + alpha + beta)
                c4 = alpha ** 2 - beta ** 2
                c5 = np.multiply(2 * (ric - 1 + alpha) * (ric - 1 + beta), (2 * (ric - 1) + alpha + beta + 2))
                L = np.multiply(c2 * (c3 * x + c4) / c1, L_1) - c5 / c1 * L_2
                L_2 = L_1
                L_1 = L
            return L
