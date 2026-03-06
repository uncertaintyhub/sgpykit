import numpy as np


def lagr_eval(current_knot, other_knots, non_grid_points):
    """
    Evaluate the Lagrange basis polynomial at non-grid points.

    Builds the Lagrange function L(x) such that:
    L(current_knot) = 1
    L(other_knots) = 0

    and returns L = L(non_grid_points)

    Parameters
    ----------
    current_knot : float
        The knot where the Lagrange function should be 1.
    other_knots : array_like
        A list or array of knots where the Lagrange function should be 0.
    non_grid_points : array_like
        The points at which to evaluate the Lagrange function.

    Returns
    -------
    L : ndarray
        The evaluated Lagrange function at non_grid_points.
    """
    # Initialize L to 1, with the same shape as non_grid_points
    L = np.ones_like(non_grid_points)

    # Iterate over each knot in other_knots
    for knots_k in other_knots:
        # Compute the Lagrange factor (x - x_k) / (current_knot - x_k)
        L *= (non_grid_points - knots_k) / (current_knot - knots_k)

    return L
