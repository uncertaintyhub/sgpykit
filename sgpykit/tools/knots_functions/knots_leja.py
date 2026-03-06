import numpy as np

from sgpykit.tools.knots_functions.constants import line_leja_tab, sym_line_leja_tab, p_disk_leja_tab


def knots_leja(n, x_a, x_b, type_, whichrho='prob'):
    """
    Compute Leja points and weights for numerical integration.

    This function returns N Leja points of a specified type between A and B,
    along with corresponding weights for numerical integration. The points and
    weights can be used to approximate integrals of the form:

    - For 'prob': int_a^b f(x) * 1/(b-a) dx
    - For 'nonprob': int_a^b f(x) dx

    Parameters
    ----------
    n : int
        Number of Leja points to generate.
    x_a : float
        Lower bound of the interval.
    x_b : float
        Upper bound of the interval.
    type_ : str
        Type of Leja points to generate. Options are:
        - 'line': Points are generated starting from B, then A, then (A+B)/2, etc.
        - 'sym_line': Points are generated symmetrically around (A+B)/2.
        - 'p_disk': Points are generated on the complex unit ball and projected to [-1,1].
    whichrho : str, optional
        Type of weight function. Options are:
        - 'prob': Probabilistic weight function (default).
        - 'nonprob': Non-probabilistic weight function.

    Returns
    -------
    x : numpy.ndarray
        Array of Leja points.
    w : numpy.ndarray
        Array of corresponding weights.

    Raises
    ------
    Exception
        If n > 31, as the precomputed tables only support up to 31 points.
    ValueError
        If the specified Leja type is unknown or if the 4th argument is not 'prob' or 'nonprob'.
    """
    # as a first step, we load the precomputed knots
    # Determine the type of Leja points and weights
    assert n>0
    if n>31:
        raise Exception('OutOfTable: too many points')
    if type_ == 'line':
        x, w = line_leja_tab[n]
    elif type_ == 'sym_line':
        x, w = sym_line_leja_tab[n]
    elif type_ == 'p_disk':
        x, w = p_disk_leja_tab[n]
    else:
        raise ValueError('unknown leja type')

    # Leja points have been precomputed on the interval (-1,1) and assuming a probabilistic weight, i.e.
    # the resulting quadrature rule is a discretization of \int_{-1}^1 f(x) 1/2 dx.
    # So now we need to rescale points and weights, if needed

    # translation to a generic interval
    if x_b != 1 or x_a != -1:
        scale_fact = (x_b - x_a) / 2
        tras = (x_b + x_a) / 2
        x = scale_fact * x + tras

    # fix weights
    if whichrho == 'prob':
        # as mentioned above, weights precomputed are already probabilistic, so in this case there's nothing to do
        pass
    elif whichrho == 'nonprob':
        # in this case, we need to rescale. The 1/2 is already included in the precomputed weights, so we only need to
        # multiply by (x_b-x_a)
        w = w * (x_b - x_a)
    else:
        raise ValueError('4th argument must be either prob or nonprob')

    x = np.transpose(x)
    w = np.transpose(w)
    return x, w
