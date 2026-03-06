import numpy as np

from sgpykit.util import matlab


def knots_exponential(n, lambda_):
    """
    Compute collocation points and weights for Gaussian integration with respect to the exponential weight function.

    This function returns the collocation points (x) and the weights (w) for Gaussian integration
    with respect to the weight function rho(x) = lambda * exp(-lambda * x), which is the density
    of an exponential random variable with rate parameter lambda.

    Parameters
    ----------
    n : int
        Number of collocation points.
    lambda_ : float
        Rate parameter of the exponential distribution.

    Returns
    -------
    x : ndarray
        Collocation points.
    w : ndarray
        Weights for Gaussian integration.
    """
    # [x,w]=KNOTS_EXPONENTIAL(n,lambda)
    # returns the collocation points (x)
    # and the weights (w) for the gaussian integration
    # w.r.t to the weight function
    # rho(x)=lambda*exp( -lambda*x )
    # i.e. the density of an exponential random variable
    # with rate parameter lambda
    assert n>0
    if n == 1:
        # the point (rescaled if needed)
        x = 1 / lambda_
        # the weight is 1:
        w = 1
        return x,w

    # calculates the values of the recursive relation
    a,b = coeflagu(n)
    # builds the matrix
    JacM = np.diag(a) + np.diag(np.sqrt(b[1:n]),1) + np.diag(np.sqrt(b[1:n]), -1)
    # calculates points and weights from eigenvalues / eigenvectors of JacM
    W,x = matlab.eig(JacM)

    w = W[0,:] ** 2
    x,ind = matlab.sort(x)

    w = w[ind]
    # modifies points according to lambda (the weigths are unaffected)
    x = x / lambda_
    #----------------------------------------------------------------------
    return x, w


def coeflagu(n):
    """
    Compute recurrence coefficients for Laguerre polynomials.

    This function returns the recurrence coefficients a and b for the Laguerre polynomials,
    which are used in the construction of the Jacobi matrix for Gaussian integration.

    Parameters
    ----------
    n : int
        Number of coefficients to compute.

    Returns
    -------
    a : ndarray
        Recurrence coefficients a.
    b : ndarray
        Recurrence coefficients b.
    """
    if n <= 1:
        raise ValueError('n must be > 1')

    a = np.zeros(n)
    b = np.zeros(n)

    a[0] = 1
    b[0] = 1

    for k in range(1, n):
        a[k] = 2 * k + 1
        b[k] = k ** 2

    return a, b
