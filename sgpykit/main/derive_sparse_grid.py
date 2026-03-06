import numpy as np

from sgpykit.main.interpolate_on_sparse_grid import interpolate_on_sparse_grid


def derive_sparse_grid(S, Sr, values_on_grid, domain, eval_points, h=None):
    """
    Compute derivatives (gradients) of a scalar-valued function f: R^N -> R
    by centered finite differences formulas applied to the sparse grid approximation of f.
    The gradients can be computed at *several* points simultaneously.

    Parameters
    ----------
    S : struct
        Sparse grid struct.
    Sr : struct
        Reduced version of S.
    values_on_grid : ndarray
        Values of the interpolated function on Sr (VALUES_ON_GRID is a vector,
        because the function is scalar-valued).
    domain : ndarray
        2xN matrix = [a1, a2, a3, ...; b1, b2, b3, ...] defining the lower and upper bound of the
        hyper-rectangle on which the sparse grid is defined.
    eval_points : ndarray
        Points where the derivative must be evaluated. It is a matrix with points stored as
        columns, following the convention of the package.
    h : ndarray or float, optional
        Finite differences increment. H can be a scalar or a vector, in which case the n-th entry
        will be used as increment to approximate the n-th component of the gradient.
        If None, the increment size is chosen according to the length of each interval [an bn] as
        h_n = (b_n - a_n)/1E5.

    Returns
    -------
    grads : ndarray
        Computed values of the derivatives (gradients). The gradients in each point are stored
        as columns of GRADS, i.e., size(GRADS) = N x size(EVAL_POINTS,2).
    """
    # get dimensions
    N = domain.shape[1]

    M = eval_points.shape[1]

    grads = np.zeros((N, M))

    a = domain[0, :]
    b = domain[1, :]
    if h is None:
        h = (b - a) / 100000.0
    elif len(h) == 1:
        h = np.full((1, N), h)

    for k in range(N):
        epsi = np.zeros((N, M))
        epsi[k, :] = h[k]  # broadcasting fills the whole row
        fintp0 = interpolate_on_sparse_grid(S, Sr, values_on_grid, eval_points + epsi)
        fintp1 = interpolate_on_sparse_grid(S, Sr, values_on_grid, eval_points - epsi)
        grads[k, :] = (fintp0 - fintp1) / (2 * h[k])

    return grads
