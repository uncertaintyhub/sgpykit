import numpy as np

from sgpykit.util.checks import is_numeric_scalar


def lagr_eval_fast(current_knot, other_knots, ok_len, non_grid_points, ng_size):
    """
    Evaluate the Lagrange basis polynomial at non-grid points.

    This function builds the Lagrange function L(x) such that L(current_knot)=1
    and L(other_knots)=0, and returns L evaluated at non_grid_points.

    Parameters
    ----------
    current_knot : float
        The knot where the Lagrange polynomial equals 1.
    other_knots : array_like
        The knots where the Lagrange polynomial equals 0.
    ok_len : int
        The length of other_knots.
    non_grid_points : array_like
        The points at which to evaluate the Lagrange polynomial.
    ng_size : int or tuple
        The size of non_grid_points. If scalar, it is treated as the size of a
        square matrix. If tuple, it is treated as the shape of non_grid_points.

    Returns
    -------
    L : ndarray
        The evaluated Lagrange polynomial at non_grid_points.
    """
    # L = LAGR_EVAL_FAST(current_knot,other_knots,ok_len,non_grid_points,ng_size)

    # builds the lagrange function L(x) s.t.
    # L(current_knot)=1;
    # L(other_knots)=0;

    # and returns L=L(non_grid_points)

    # where ok_len = length(other_knots), ng_size = size(non_grid_points);

    # this is essentially the same function as LAGR_EVAL, but some quantities are provided as input
    # instead of being computed, for speed purposes.
    # each monodim lagrange function is a product of K terms like (x-x_k)/(current_knot-x_k),
    # so i compute it with a for loop

    # this is the number of iterations
    # ok_len=length(other_knots);

    # L is the result. It is a column vector, same size as non_grid_points. It is initialized to 1, and then iteratively multiplied by
    # each factor
    # L=ones(size(non_grid_points));
    if is_numeric_scalar(ng_size):
        L = np.ones((ng_size, ng_size))
    else:
        # make a simple vector (n,) instead of a column vector (n,1)
        if len(ng_size) == 2 and ng_size[1] == 1:
            ng_size = ng_size[0]
        L = np.ones(ng_size)

    for k in range(ok_len):
        # these are the non current-knot, one at a time
        knots_k = other_knots[k]
        # here it comes the computation of the lagrangian factor (x-x_k)/(current_knot-x_k)
        L *= (non_grid_points - knots_k) / (current_knot - knots_k)

    return L
