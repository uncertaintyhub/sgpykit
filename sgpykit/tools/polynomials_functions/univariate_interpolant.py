import numpy as np

from sgpykit.tools.polynomials_functions.lagr_eval import lagr_eval


def univariate_interpolant(grid_points, function_on_grid, non_grid_points):
    """
    Interpolates values at non_grid_points using a Lagrange interpolant
    built from given grid_points and tabulated function_on_grid.

    Parameters
    ----------
    grid_points : (M,) array_like
        Interpolation nodes.
    function_on_grid : (V, M) array_like
        Values of V different functions sampled at the M grid points.
    non_grid_points : (N,) array_like
        Points where we want to evaluate the interpolation.

    Returns
    -------
    f_values : (V, N) ndarray
        Interpolated values at non_grid_points.

    Notes
    -----
    UNIVARIATE_INTERPOLANT interpolates a univariate function on grid points, i.e. evaluates the
    Lagrange polynomial approximation on a generic point of the parameter space.

    F_VALUES = UNIVARIATE_INTERPOLANT(GRID_POINTS,FUNCTION_ON_GRID,NON_GRID_POINTS) evaluates the
        lagrangian interpolant of a vector function F: R -> R^V based on the points contained in the vector GRID_POINTS.

        GRID POINTS is a row vector with the nodes of interpolation.
        FUNCTION_ON_GRID is a matrix containing the evaluation of F on the points GRID_POINTS.
            Its dimensions are V x length(GRID_POINTS).
        NON_GRID_POINTS is a row vector of points where one wants to evaluate the polynomial approximation.
        F_VALUES is a matrix containing the evaluation of the function F in each of the NON_GRID_POINTS.
            Its dimensions are V X length(NON_GRID_POINTS)
    """
    V, nb_grid_points = function_on_grid.shape
    if len(grid_points) != nb_grid_points:
        raise ValueError('the number of function evaluations does not match the number of interpolation knots')

    grid_points = np.asarray(grid_points)
    function_on_grid = np.asarray(function_on_grid)
    non_grid_points = np.asarray(non_grid_points)

    V, nb_grid_points = function_on_grid.shape
    nb_pts = non_grid_points.size

    f_values = np.zeros((V, nb_pts), dtype=float)

    for k, xk in enumerate(grid_points):
        # Build all basis values efficiently
        Lk = lagr_eval(xk, np.delete(grid_points, k), non_grid_points)
        f_values += function_on_grid[:, k][:, None] * Lk[None, :]

    return f_values
