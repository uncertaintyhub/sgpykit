import numpy as np
from scipy.special import gamma
from sgpykit.util import matlab


def knots_beta(n, alpha, beta, x_a, x_b):
    """
    Calculate the collocation points (x) and weights (w) for Gaussian integration
    with respect to the Beta distribution weight function.

    The weight function is defined as:
    rho(x) = Gamma(alpha+beta+2)/(Gamma(alpha+1)*Gamma(beta+1)*(x_b-x_a)^(alpha+beta+1)) *
             (x-x_a)^alpha * (x_b-x)^beta

    This corresponds to the density of a Beta random variable with range [x_a, x_b]
    and parameters alpha, beta > -1.

    Parameters
    ----------
    n : int
        Number of collocation points.
    alpha : float
        First parameter of the Beta distribution.
    beta : float
        Second parameter of the Beta distribution.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.

    Returns
    -------
    x : ndarray
        Array of collocation points.
    w : ndarray
        Array of corresponding weights.

    Notes
    -----
    For n=1, the standard node and weight are returned directly.
    For n>1, the points and weights are derived from the eigenvalues and eigenvectors
    of the Jacobi matrix constructed from recurrence coefficients.
    """
    assert n > 0
    if n == 1:
        # standard node
        x = (beta - alpha) / (alpha + beta + 2)
        # the weight is 1:
        w = 1
    else:
        # calculates the values of the recursive relation
        a, b = coefjac(n, alpha, beta)
        # builds the matrix
        JacM = np.diag(a) + np.diag(np.sqrt(b[1:n]), 1) + np.diag(np.sqrt(b[1:n]), -1)
        # calculates points and weights from eigenvalues / eigenvectors of JacM
        W, x = matlab.eig(JacM)

        w = W[0,:] ** 2
        x, ind = matlab.sort(x)
        w = w[ind]

    # modifies points according to x_a and x_b (the weights are unaffected)
    x = ((x_a + x_b) - (x_b - x_a) * x) / 2
    # ----------------------------------------------------------------------
    return x, w


def coefjac(n, alpha, beta):
    """
    Compute recurrence coefficients for Jacobi polynomials.

    Parameters
    ----------
    n : int
        Number of coefficients to compute.
    alpha : float
        First parameter of the Jacobi polynomial.
    beta : float
        Second parameter of the Jacobi polynomial.

    Returns
    -------
    a : ndarray
        Array of diagonal coefficients.
    b : ndarray
        Array of off-diagonal coefficients.

    Raises
    ------
    ValueError
        If n is less than or equal to 1.

    Notes
    -----
    The recurrence coefficients are used to construct the Jacobi matrix for Gaussian
    quadrature. The coefficients are computed using the following formulas:

    For k=0:
        a[0] = (beta - alpha) / (alpha + beta + 2)
        b[0] = 2^(alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)

    For k=1:
        a[1] = (beta^2 - alpha^2) / ((2 + alpha + beta) * (2 + alpha + beta + 2))
        b[1] = 4 * (alpha + 1) * (beta + 1) / ((2 + alpha + beta)^2 * (alpha + beta + 3))

    For k >= 2:
        a[k] = (beta^2 - alpha^2) / ((2*k + alpha + beta - 2) * (2*k + alpha + beta))
        b[k] = 4*k*(k + alpha - 1)*(k + beta - 1)*(k + alpha + beta - 1) /
               ((2*k + alpha + beta - 2)^2 * (2*k + alpha + beta - 1) * (2*k + alpha + beta))
    """
    if n <= 1:
        raise ValueError('n must be > 1')

    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = (beta - alpha) / (alpha + beta + 2)
    b[0] = 2 ** (alpha + beta + 1) * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 2)

    a[1] = (beta ** 2 - alpha ** 2) / ((2 + alpha + beta) * (2 + alpha + beta + 2))
    b[1] = 4 * (alpha + 1) * (beta + 1) / ((2 + alpha + beta) ** 2 * (alpha + beta + 3))

    k = np.arange(2, n)
    a[k] = (beta ** 2 - alpha ** 2) / ((2 * k + alpha + beta - 2) * (2 * k + alpha + beta))
    b[k] = (4 * k * (k + alpha - 1) * (k + beta - 1) * (k + alpha + beta - 1)) / (
                (2 * k + alpha + beta - 2) ** 2 * (2 * k + alpha + beta - 1) * (2 * k + alpha + beta))

    return a, b
