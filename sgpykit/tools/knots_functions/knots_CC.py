import numpy as np

from sgpykit.util import matlab


def knots_CC(nn, x_a, x_b, whichrho='prob'):
    """
    Calculate the collocation points and weights for the Clenshaw-Curtis integration formula.

    This function computes the collocation points (x) and weights (w) for the Clenshaw-Curtis
    integration formula with respect to a specified weight function. The weight function can be
    either a uniform density (rho(x) = 1/(x_b - x_a)) or a uniform weight (rho(x) = 1).

    Parameters
    ----------
    nn : int
        Number of collocation points. Must be an odd number for efficiency reasons.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.
    whichrho : str, optional
        Specifies the weight function. Options are:
        - 'prob' : Uniform density (default).
        - 'nonprob' : Uniform weight.

    Returns
    -------
    x : numpy.ndarray
        Array of collocation points.
    w : numpy.ndarray
        Array of weights corresponding to the collocation points.

    Raises
    ------
    ValueError
        If `nn` is not an odd number or if `whichrho` is not recognized.
    """
    # [x,w] = KNOTS_CC(nn,x_a,x_b)
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1/(x_b-x_a)
    # i.e. the density of a uniform random variable
    # with range going from x=x_a to x=x_b. Note that for efficiency reasons
    # nn must be an odd number
    # [x,w] = KNOTS_CC(nn,x_a,x_b,'prob')
    # is the same as [x,w] = KNOTS_CC(nn,x_a,x_b) above
    # [x,w]=[x,w] = KNOTS_CC(nn,x_a,x_b,'nonprob')
    # calculates the collocation points (x)
    # and the weights (w) for the Clenshaw-Curtis integration formula
    # w.r.t to the weight function rho(x)=1
    assert nn>0
    if nn == 1:
        x = (x_a + x_b) / 2
        wt = 1
    elif np.mod(nn, 2) == 0:
        raise ValueError('error in knots_CC: Clenshaw-Curtis formula. Use only odd number of points')
    else:
        n = nn - 1
        N = np.arange(1, n, 2)
        l = end_N = N.shape[0]
        m = n - l
        v0 = np.concatenate((2. / N / (N - 2.), [1. / N[end_N - 1]], np.zeros(m)))
        end_v0 = v0.shape[0]
        v2 = -v0[0:end_v0 - 1] - v0[end_v0 - 1:0:-1]

        g0 = -np.ones(n)
        g0[l] = g0[l] + n
        g0[m] = g0[m] + n
        g = g0 / (n ** 2 - 1 + n % 2)

        wcc = np.real(matlab.ifft(v2 + g))
        wt = np.concatenate((wcc, [wcc[0]])) / 2.

        x = np.cos(np.arange(0, n + 1) * np.pi / n)
        x = ((x_b - x_a) / 2.) * x + (x_a + x_b) / 2.

    if 'nonprob' == whichrho:
        w = (x_b - x_a) * wt
    else:
        if 'prob' == whichrho:
            w = wt
        else:
            raise ValueError('whichrho input not recognized')

    return x, w
