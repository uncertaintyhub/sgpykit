import numpy as np

from sgpykit.util import matlab


def knots_uniform(n, x_a, x_b, whichrho=None):
    """
    Calculate collocation points and weights for Gaussian integration with respect to a uniform weight function.

    This function computes the collocation points (x) and weights (w) for Gaussian integration
    with respect to the weight function rho(x) = 1/(b-a), which corresponds to the density of a
    uniform random variable on the interval [x_a, x_b].

    Parameters
    ----------
    n : int
        Number of collocation points.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.
    whichrho : str, optional
        Type of weight function. Options are:
        - 'prob' (default): Weight function rho(x) = 1/(b-a).
        - 'nonprob': Weight function rho(x) = 1.

    Returns
    -------
    x : ndarray
        Array of collocation points.
    w : ndarray
        Array of corresponding weights.

    Notes
    -----
    For n=1, the collocation point is the midpoint of the interval, and the weight is 1.
    For n>1, the collocation points and weights are computed using the eigenvalues and eigenvectors
    of the Jacobi matrix derived from the recurrence coefficients of the Legendre polynomials.
    """
    assert n > 0

    if whichrho is None:
        whichrho = 'prob'

    if n == 1:
        x = (x_a + x_b) / 2
        wt = 1
    else:
        # calculates the values of the recursive relation
        a, b = coeflege(n)
        # Create the Jacobi matrix
        JacM = np.diag(a) + np.diag(np.sqrt(b[1:n]), k=1) + np.diag(np.sqrt(b[1:n]), k=-1)

        # Compute eigenvalues and eigenvectors
        W, x = matlab.eig(JacM)

        # Extract eigenvalues
        x = x.real  # Use .real to ensure real part if eigenvalues are complex

        # Compute weights
        wt = W[0, :] ** 2

        # Sort eigenvalues and corresponding weights
        ind = np.argsort(x)
        x = x[ind]
        wt = wt[ind]

        # modifies points according to the distribution and its interval x_a, x_b
        x = (x_b - x_a) / 2 * x + (x_a + x_b) / 2

    # finally, fix weights

    if 'nonprob' == whichrho:
        w = (x_b - x_a) * wt
    else:
        if 'prob' == whichrho:
            w = wt
        else:
            raise ValueError('4th input not recognized')

    # ----------------------------------------------------------------------
    return x, w


def coeflege(n):
    """
    Compute recurrence coefficients for Legendre polynomials.

    This function returns the recurrence coefficients a and b for the Legendre polynomials,
    which are used to construct the Jacobi matrix for Gaussian quadrature.

    Parameters
    ----------
    n : int
        Order of the recurrence coefficients.

    Returns
    -------
    a : ndarray
        Array of recurrence coefficients a.
    b : ndarray
        Array of recurrence coefficients b.

    Notes
    -----
    The recurrence coefficients are computed as follows:
    - a is an array of zeros of length n.
    - b[0] = 2.
    - For k >= 1, b[k] = 1 / (4 - 1 / (k ** 2)).
    """
    # if (n <= 1):
    #     raise Exception('n must be > 1')

    a = np.zeros(n)
    b = np.zeros(n)

    b[0] = 2

    k = np.arange(1, n)
    b[k] = 1 / (4 - 1 / (k ** 2))
    return a, b
