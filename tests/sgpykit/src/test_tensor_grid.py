import pytest
import numpy as np

from sgpykit.src import tensor_grid
from sgpykit.tools.knots_functions.knots_CC import knots_CC
from sgpykit.tools.lev2knots_functions.lev2knots_doubling import lev2knots_doubling
from sgpykit.util import matlab
from sgpykit.util.checks import contains_same_elements, is_list_math_equal


def test_tensor_grid():
    #knots=@(n) knots_CC(n,-1,1,'nonprob'); % knots
    #lev2knots = @lev2knots_doubling
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    N = 3
    ii = np.array([1, 2, 3])
    m = lev2knots_doubling(ii)
    s = tensor_grid(N, m, knots)

    assert len(s) == 1  # tensor grid is a scalar StructArray
    assert contains_same_elements(matlab.fieldnames(s), {'knots', 'weights', 'size', 'knots_per_dim', 'm'})

    prec = 15
    val = 2**(-0.5)
    exp_knots = np.round(np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.0000, 0.0000, -1.0000, 1.0000, 0.0000, -1.0000, 1.0000, 0.0000,
             -1.0000, 1.0000, 0.0000, -1.0000, 1.0000, 0.0000, -1.0000],
            [1.0000, 1.0000, 1.0000, val, val, val, 0.0000, 0.0000,
             0.0000, -val, -val, -val, -1.0000, -1.0000, -1.0000]
        ]), prec)
    result = np.round(np.array(s.knots), prec)
    np.testing.assert_array_almost_equal(result, exp_knots)

    exp_weights = np.array([[
        0.044444, 0.177778, 0.044444, 0.355556, 1.422222, 0.355556, 0.533333,
        2.133333, 0.533333, 0.355556, 1.422222, 0.355556, 0.044444, 0.177778,
        0.044444
    ]])
    np.testing.assert_array_almost_equal(s.weights, exp_weights)

    exp_knotsperdim = [[0],
                       [1.0, 0, -1.0],
                       [1.0, val, 0, -val, -1.0]]
    assert is_list_math_equal(s.knots_per_dim, exp_knotsperdim)

    exp_m = [1, 3, 5]
    assert is_list_math_equal(s.m, exp_m)