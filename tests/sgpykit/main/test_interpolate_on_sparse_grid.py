import pytest
import numpy as np
import sgpykit as sg


def test_interpolate_on_sparse_grid():
    # Define the function f
    f = lambda x, b: np.prod(1 / np.sqrt(x + b), axis=0)
    # Parameters
    b = 3
    N = 3

    # Define the number of levels
    w = 4

    # Define the knots function
    knots = lambda n: sg.knots_uniform(n, -1, 1, 'nonprob')

    # Create the sparse grid
    S,_ = sg.create_sparse_grid(N, w, knots, sg.lev2knots_lin)

    # Reduce the sparse grid
    Sr = sg.reduce_sparse_grid(S)

    # Define non-grid points
    # non_grid_points = np.random.rand(N, 100)
    non_grid_points = np.hstack((0.5 * np.ones((N, 1)), np.zeros((N, 1))))

    # Compute the nodal values to be used to interpolate
    function_on_grid = f(Sr.knots, b)

    # Compute interpolated values
    f_values = sg.interpolate_on_sparse_grid(S, Sr, function_on_grid, non_grid_points)

    # Compute the interpolation error
    interpolation_error = np.max(np.abs(f_values - f(non_grid_points, b)))

    assert np.isclose(interpolation_error, 7.0684e-06)


def test_interpolate_on_sparse_grid_random():
    assert True
    N = 3
    nr = 10 # number random non-grid points
    # Define the functions f1, f2, and f using lambdas
    f1 = lambda x: 1 / (1 + 0.5 / N * np.sum(x, axis=0))
    f2 = lambda x: np.sum(np.abs(x) ** 3, axis=0)

    f = lambda x: np.vstack((f1(x), f2(x)))

    # Define sparse grid
    lev2knots, idxset = sg.define_functions_for_rule('TD', N)
    knots = lambda n: sg.knots_uniform(n, -1, 1, 'nonprob')

    # Initialize variables
    w_max = 1
    interp_error = np.zeros((2, w_max + 1))
    work = np.zeros(w_max + 1)
    pol_size = np.zeros(w_max + 1)

    # Generate non-grid points (from matlab example)
    non_grid_points = np.array([
        [-4.954652340488994e-01, -8.766135709659397e-01,  6.449036412344840e-01, -2.215420044954466e-01,  5.445040242462174e-01, -1.325279487496567e-02, -9.407143784518828e-01,  3.803785933211468e-02,  2.650478674257299e-01, -6.317634489391670e-01],
        [-3.199353375058556e-01, -8.605745350528644e-01, -2.953668492152326e-01, -8.133440336561901e-01,  7.180020940348153e-02,  5.238717732522942e-01,  6.037952793684878e-01,  2.587218467620418e-01, -3.695104041697661e-01,  1.209117453234032e-01],
        [ 7.858001296515880e-01, -1.501538180333859e-01,  8.821821084882817e-01,  9.527120797712230e-01, -4.781343952598192e-01,  3.383793695403425e-02, -7.926943559291721e-01,  8.865404131241055e-01, -2.254014005317144e-01,  8.556596599913411e-01]
    ])

    S_old = None  # we also recycle previous grids

    # For loop
    for w in range(w_max+1):
        # Create grid
        S, C = sg.create_sparse_grid(N, w, knots, lev2knots, idxset, S_old)
        S_old = S

        Sr = sg.reduce_sparse_grid(S)

        # Compute work
        work[w] = Sr.size

        # Compute estimate of polynomial size
        pol_size[w] = C.shape[0]

        # Compute the nodal values to be used to interpolate. It has to be
        # row vector (more rows for vector-valued output functions)
        function_on_grid = f(Sr.knots)

        # Compute interpolated values. Here f_values is row
        f_values = sg.interpolate_on_sparse_grid(S, Sr, function_on_grid, non_grid_points)

        # Compute error
        tmp = np.max(np.abs(f(non_grid_points) - f_values), axis=1)
        interp_error[:, w] = tmp

    exp = np.array([[0.458910487864282,0.113376332416250], [1.550704865070245, 0.973354595880619]])
    assert np.allclose(exp, interp_error)