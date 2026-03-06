import numpy as np

from sgpykit.main.interpolate_on_sparse_grid import interpolate_on_sparse_grid


def hessian_sparse_grid(S, Sr, values_on_grid, domain, eval_point, h=None):
    """
    Compute the Hessian of a scalar-valued function f: R^N -> R by finite differences
    applied to the sparse grids approximation of f.

    This function can only be evaluated at a single point, unlike derive_sparse_grid.

    Parameters
    ----------
    S : struct
        Sparse grid struct.
    Sr : struct
        Reduced version of S.
    values_on_grid : ndarray
        Values of the interpolated function on Sr.
    domain : ndarray
        2xN matrix = [a1, a2, a3, ...; b1, b2, b3, ...] defining the lower and upper
        bound of the hyper-rectangle on which the sparse grid is defined.
    eval_point : ndarray
        Column vector point where the derivative must be evaluated.
    h : scalar or ndarray, optional
        Finite differences increment size. If None, the increment size is chosen according
        to the length of each interval [an bn] as h_n = (b_n - a_n)/1E5.

    Returns
    -------
    Hessian : ndarray
        Hessian matrix, H(i,j) = \\partial_{y_i} \\partial_{y_j} sparse grid.
    """
    # get dimensions
    N = domain.shape[1]  # the sparse grid is defined over an N-variate hypercube

    Hessian = np.zeros((N, N))

    if h is None:
        a = domain[0, :]
        b = domain[1, :]
        h = (b - a) / 1E5
    elif np.isscalar(h):
        h = np.full((1, N), h)
    if eval_point.ndim == 1: # convert it to a column vector then
        eval_point = np.atleast_2d(eval_point).T

    # the evaluation of the sparse grids at the point (i.e. the center of the stencil) can be done once and for all
    f_0 = interpolate_on_sparse_grid(S, Sr, values_on_grid, eval_point)

    for j in range(N):

        # -------------------------------------------------------
        # the extra-diagonal part of the matrix (upper part)
        for k in range(j + 1, N):
            # the formula for D_jk uses 4 points: (only values of j and k index are specified)
            # ( f_{j+1,k+1} - f_{j-1,k+1} - f_{j+1,k-1} + f_{j-1,k-1} ) / (4 h_j h_k)

            epsi = np.zeros((N, 4))

            # the first point is beta + he_j + he_k
            epsi[[j, k], 0] = [h[j], h[k]]
            # the second point is beta - he_j + he_k
            epsi[[j, k], 1] = [-h[j], h[k]]
            # the third point is beta + he_j - he_k
            epsi[[j, k], 2] = [h[j], -h[k]]
            # the fourth point is beta -he_j - he_k
            epsi[[j, k], 3] = [-h[j], -h[k]]

            # the evaluation
            f_evals = interpolate_on_sparse_grid(S, Sr, values_on_grid, eval_point + epsi)
            # the approx of Hessian entry
            Hessian[j, k] = (f_evals.ravel() @ np.array([1, -1, -1, 1])) / (4 * h[j] * h[k])

        # -------------------------------------------------------
        # the extra-diagonal part of the matrix (lower part) is copied from the upper part that we have already computed
        for k in range(j):
            Hessian[j, k] = Hessian[k, j]

        # -------------------------------------------------------
        # then the diagonal entry
        # the formula for D_jj uses 3 points: (only values of j and k index are specified)
        # ( f_{j+1} - 2*f_{j} + f_{j-1} ) / h_j^2
        #
        # note that we already computed f_j once and for all before the loop (variable f_0)

        epsi = np.zeros((N, 2))

        # the first point is beta + he_j
        epsi[j, 0] = h[j]
        # the second point is beta - he_j
        epsi[j, 1] = -h[j]

        # the evaluation
        f_evals = interpolate_on_sparse_grid(S, Sr, values_on_grid, eval_point + epsi)
        # the approx of Hessian entry
        Hessian[j, j] = (np.sum(f_evals) - 2 * f_0[0,0]) / h[j] ** 2

    return Hessian
