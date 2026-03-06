import numpy as np

from sgpykit.src.GK_lev_table import GK_lev_table
from sgpykit.tools.knots_functions.constants import GK_tabulated
from sgpykit.util import matlab


def knots_GK(n, mi, sigma):
    """
    Compute collocation points and weights for Genz-Keister quadrature.

    This function returns the collocation points (x) and the weights (w) of the
    Genz-Keister quadrature formula (also known as Kronrod-Patterson-Normal) for
    approximation of integrals with respect to the weight function:
    rho(x) = 1/sqrt(2*pi*sigma^2) * exp(-(x-mi)^2 / (2*sigma^2)),
    i.e., the density of a Gaussian random variable with mean mi and standard
    deviation sigma.

    Knots and weights have been precomputed (up to 35). An error is raised if the
    number of points requested is not available.

    Parameters
    ----------
    n : int
        Number of collocation points.
    mi : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    x : ndarray
        Collocation points.
    w : ndarray
        Weights corresponding to the collocation points.

    References
    ----------
    Florian Heiss, Viktor Winschel,
    Likelihood approximation by numerical integration on sparse grids,
    Journal of Econometrics,
    Volume 144, 2008, pages 62-80.

    Alan Genz, Bradley Keister,
    Fully symmetric interpolatory rules for multiple integrals
    over infinite regions with Gaussian weight,
    Journal of Computational and Applied Mathematics,
    Volume 71, 1996, pages 299-309.
    """
    # first recover the i-level
    assert n > 0
    i = knots2lev_GK(n)
    if np.isnan(i):
        raise Exception(f"OutOfTable: this number of points is not available: {n}")
    else:
        # now access the knots and weights table using the i2l map
        x_t, w_t = GK_tabulated[GK_lev2l_map(i)]
        x = redistribute_knots(x_t)
        w = redistribute_weights(w_t)

    x = mi + sigma * x
    # sort knots increasingly and weights accordingly
    x, sorter = matlab.sort(x)
    w = w[sorter]
    return x, w


def knots2lev_GK(nb_knots):
    """
    Map the number of knots to the corresponding level.

    Parameters
    ----------
    nb_knots : int
        Number of knots.

    Returns
    -------
    int or float
        Corresponding level or np.nan if not found.
    """
    # Map the number of knots to the corresponding level
    knot_to_level = {
        0: GK_lev_table(0, 0),
        1: GK_lev_table(1, 0),
        3: GK_lev_table(2, 0),
        9: GK_lev_table(3, 0),
        19: GK_lev_table(4, 0),
        35: GK_lev_table(5, 0)
    }
    # Return the corresponding level or np.nan if not found
    return knot_to_level.get(nb_knots, np.nan)


def GK_lev2l_map(i):
    """
    Map the condensed level to the true level.

    Parameters
    ----------
    i : int
        Condensed level.

    Returns
    -------
    int
        True level corresponding to the condensed level.
    """
    # l = GK_lev2l_map(i)
    # returns the l-level (``true level'') corresponding to the level i (``condensed level''),
    # as from table in GK_lev_table

    if i > 5:
        raise Exception('OutOfTable: this level is not tabulated')
    else:
        # i to l map
        l = GK_lev_table(i, 1)

    return l


def redistribute_knots(x):
    """
    Redistribute knots to form a symmetric set.

    Parameters
    ----------
    x : array_like
        Input knots.

    Returns
    -------
    ndarray
        Redistributed knots.
    """
    arr = np.asarray(x).ravel()
    n = arr.size
    if n <= 1:
        return arr.copy()
    rest = arr[1:]  # everything except the first element
    out_len = 1 + 2 * rest.size
    out = np.empty(out_len, dtype=arr.dtype)

    out[0] = arr[0]  # first element stays unchanged
    out[1::2] = -rest  # positions 1,3,5...
    out[2::2] = rest  # positions 2,4,6...
    return out


def redistribute_weights(x):
    """
    Redistribute weights to match the redistributed knots.

    Parameters
    ----------
    x : array_like
        Input weights.

    Returns
    -------
    ndarray
        Redistributed weights.
    """
    a = np.asarray(x).ravel()
    if a.size <= 1:
        return a.copy()
    rest = a[1:]
    out = np.empty(1 + 2 * rest.size, dtype=a.dtype)
    out[0] = a[0]
    out[1::2] = rest
    out[2::2] = rest
    return out
