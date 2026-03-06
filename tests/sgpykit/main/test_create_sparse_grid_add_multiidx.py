import pytest
import numpy as np

import sgpykit as sg
from sgpykit.util import matlab
from sgpykit.util.checks import is_list_math_equal
from sgpykit.util.misc import matlab_to_python_index


def test_create_sparse_grid_add_multiidx():
    a, b = -1.0, 1.0
    # knot generator (Clenshaw‑Curtis on [a,b])
    knots = lambda n: sg.knots_CC(n, a, b)

    # level‑to‑knots mapping (doubling rule)
    lev2knots = sg.lev2knots_doubling
    jj = np.array([0,3])
    G = matlab_to_python_index(np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1]]))
    coeff_G = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0]
    S = matlab.struct(
        knots=
        matlab.ce(np.array([[1.000000e+00, 6.123234e-17, -1.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00]]),
                  np.array([[1.00000000e+00, 6.12323400e-17, -1.00000000e+00,
                             1.00000000e+00, 6.12323400e-17, -1.00000000e+00,
                             1.00000000e+00, 6.12323400e-17, -1.00000000e+00,
                             1.00000000e+00, 6.12323400e-17, -1.00000000e+00,
                             1.00000000e+00, 6.12323400e-17, -1.00000000e+00],
                            [1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                             7.07106781e-01, 7.07106781e-01, 7.07106781e-01,
                             6.12323400e-17, 6.12323400e-17, 6.12323400e-17,
                             -7.07106781e-01, -7.07106781e-01, -7.07106781e-01,
                             -1.00000000e+00, -1.00000000e+00, -1.00000000e+00]]),
                  np.array([[1.00000000e+00, 7.07106781e-01, 6.12323400e-17,
                             -7.07106781e-01, -1.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00]])),
        weights=matlab.ce(np.array([[-0.16666667, -0.66666667, -0.16666667]]),
                          np.array([[0.00555556, 0.02222222, 0.00555556, 0.04444444, 0.17777778, 0.04444444,
                                     0.06666667, 0.26666667, 0.06666667, 0.04444444, 0.17777778, 0.04444444,
                                     0.00555556, 0.02222222, 0.00555556]]),
                          np.array([[0.03333333, 0.26666667, 0.4, 0.26666667, 0.03333333]])),
        size=matlab.ce(3, 15, 5),
        knots_per_dim=matlab.ce([np.array([1.0, 6.123233995736766e-17, -1.0]), np.array([0.0])],
                                [np.array([1.0, 6.123233995736766e-17, -1.0]),
                                 np.array([1.0, 0.7071067811865476, 6.123233995736766e-17, -0.7071067811865475, -1.0])],
                                [np.array([1.0, 0.7071067811865476, 6.123233995736766e-17, -0.7071067811865475, -1.0]),
                                 np.array([0.0])]),
        m=matlab.ce([3, 1], [3, 5], [5, 1]),
        coeff=matlab.ce(-1.0, 1.0, 1.0),
        idx=matlab.ce([2, 1], [2, 3], [3, 1])
    )

    T, G, coeff_G = sg.create_sparse_grid_add_multiidx(jj, S, G, coeff_G, knots, lev2knots)
    exp_G = matlab_to_python_index(np.array([[1, 1],
       [1, 2],
       [1, 3],
       [1, 4],
       [2, 1],
       [2, 2],
       [2, 3],
       [3, 1]]))
    exp_coeff_G = np.array([0., 0., -1., 1., -1., 0., 1., 1.])
    exp_T_size = [5,9,3,15,5]
    exp_T_idx = matlab_to_python_index(np.array([[1, 3], [1, 4], [2, 1], [2, 3], [3, 1]]))
    exp_T_coeff = [-1., 1., -1., 1., 1.]
    exp_T_weights = [
        [[-0.03333333, -0.26666667, -0.4       , -0.26666667, -0.03333333]],
        [[0.00793651, 0.07310932, 0.13968254, 0.18085893, 0.1968254 ,
        0.18085893, 0.13968254, 0.07310932, 0.00793651]],
        [[-0.16666667, -0.66666667, -0.16666667]],
        [[0.00555556, 0.02222222, 0.00555556, 0.04444444, 0.17777778,
        0.04444444, 0.06666667, 0.26666667, 0.06666667, 0.04444444,
        0.17777778, 0.04444444, 0.00555556, 0.02222222, 0.00555556]],
        [[0.03333333, 0.26666667, 0.4       , 0.26666667, 0.03333333]]]
    assert np.all(exp_G == G)
    assert np.all(exp_coeff_G == coeff_G)
    assert np.all(exp_T_size == T.size)
    assert is_list_math_equal(exp_T_idx, T.idx)
    assert np.allclose(exp_T_coeff, T.coeff)
    assert is_list_math_equal(exp_T_weights, T.weights, tol=1e-6)