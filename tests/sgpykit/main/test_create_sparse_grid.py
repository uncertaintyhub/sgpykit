import pytest
import numpy as np

from sgpykit import knots_uniform
from sgpykit.main import create_sparse_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling, lev2knots_lin
from sgpykit.util.checks import is_list_math_equal
from sgpykit.util.misc import matlab_to_python_index


def test_create_sparse_grid():
    N = 2  # do not change
    w = 2  # do not change
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    # expected
    knots_expected = [
    [
        [0, 0, 0],
        [1.0000, 0.0000, -1.0000]
    ],
    [
        [0, 0, 0, 0, 0],
        [1.0000, 0.7071, 0.0000, -0.7071, -1.0000]
    ],
    [
        [1.0000, 0.0000, -1.0000],
        [0, 0, 0]
    ],
    [
        [1.0000e+00, 6.1232e-17, -1.0000e+00, 1.0000e+00, 6.1232e-17, -1.0000e+00, 1.0000e+00, 6.1232e-17, -1.0000e+00],
        [1.0000e+00, 1.0000e+00, 1.0000e+00, 6.1232e-17, 6.1232e-17, 6.1232e-17, -1.0000e+00, -1.0000e+00, -1.0000e+00]
    ],
    [
        [1.0000, 0.7071, 0.0000, -0.7071, -1.0000],
        [0, 0, 0, 0, 0]
    ]]
    weights_expected = [
        [
            [-0.666666666666667, -2.666666666666667, -0.666666666666667]
        ],
        [
            [0.133333333333333, 1.066666666666667, 1.600000000000000, 1.066666666666667, 0.133333333333333]
        ],
        [
            [-0.666666666666667, -2.666666666666667, -0.666666666666667]
        ],
        [
            [0.111111111111111, 0.444444444444445, 0.111111111111111,
            0.444444444444445, 1.777777777777778, 0.444444444444445,
            0.111111111111111, 0.444444444444445, 0.111111111111111]
        ],
        [
            [0.133333333333333, 1.066666666666667, 1.600000000000000,
            1.066666666666667, 0.133333333333333]
        ]
    ]
    size_expected = np.array([3,5,3,9,5])
    knots_per_dim_expected = [
    [
        [0],
        [1.000000000000000e+00, 6.123233995736766e-17, -1.000000000000000e+00]
    ],
    [
        [0],
        [1.000000000000000e+00, 7.071067811865476e-01, 6.123233995736766e-17, -7.071067811865475e-01, -1.000000000000000e+00]
    ],
    [
        [1.000000000000000e+00, 6.123233995736766e-17, -1.000000000000000e+00],
        [0]
    ],
    [
        [1.000000000000000e+00, 6.123233995736766e-17, -1.000000000000000e+00],
        [1.000000000000000e+00, 6.123233995736766e-17, -1.000000000000000e+00]
    ],
    [
        [1.000000000000000e+00, 7.071067811865476e-01, 6.123233995736766e-17, -7.071067811865475e-01, -1.000000000000000e+00],
        [0]
    ]]
    m_expected = np.array([
        [1, 3],
        [1, 5],
        [3, 1],
        [3, 3],
        [5, 1]
    ])
    coeff_expected = np.array([-1, 1, -1, 1, 1])
    idx_expected = matlab_to_python_index(np.array([
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [3, 1]
    ]))
    assert is_list_math_equal(S.knots, knots_expected, tol=1e-4) # TODO: get more precise results
    assert is_list_math_equal(S.weights, weights_expected)
    assert is_list_math_equal(S.size, size_expected)
    assert is_list_math_equal(S.knots_per_dim, knots_per_dim_expected)
    assert is_list_math_equal(S.m, m_expected)
    assert is_list_math_equal(S.coeff, coeff_expected)
    assert is_list_math_equal(S.idx, idx_expected)


def test_create_sparse_grid_shift():
    N = 2  # do not change
    w = 1  # do not change
    knots = lambda n: knots_CC(n, -2, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    # expected
    knots_expected = [
        [
            [-0.5],
            [-0.5],
        ],[
            [-0.5, -0.5, -0.5],
            [1.0, -0.5, -2.0]
        ],[
            [1.0, -0.5, -2.0],
            [-0.5, -0.5, -0.5]
        ]]
    m_expected = [[1,1],[1,3],[3,1]]
    weights_expected = [
        [
            [-9.0],
        ],[
            [1.5, 6.0, 1.5],
        ],[
            [1.5, 6.0, 1.5]
        ]]
    size_expected = [1,3,3]
    idx_expected = matlab_to_python_index(np.array([[1,1],[1,2],[2,1]]))
    coeff_expected = [-1,1,1]
    knots_per_dim_expected = [
        [
            [-0.5],
            [-0.5],
        ],[
            [-0.5],
            [1.0, -0.5, -2.0]
        ],[
            [1.0, -0.5, -2.0],
            [-0.5]
        ]]
    assert is_list_math_equal(S.knots, knots_expected)
    assert is_list_math_equal(S.weights, weights_expected)
    assert is_list_math_equal(S.size, size_expected)
    assert is_list_math_equal(S.knots_per_dim, knots_per_dim_expected)
    assert is_list_math_equal(S.m, m_expected)
    assert is_list_math_equal(S.coeff, coeff_expected)
    assert is_list_math_equal(S.idx, idx_expected)


def test_create_sparse_grid_shift_prob():
    N = 2  # do not change
    w = 1  # do not change
    knots = lambda n: knots_CC(n, -2, 1, 'prob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    # expected
    knots_expected = [
        [
            [-0.5],
            [-0.5],
        ],[
            [-0.5, -0.5, -0.5],
            [1.0, -0.5, -2.0]
        ],[
            [1.0, -0.5, -2.0],
            [-0.5, -0.5, -0.5]
        ]]
    m_expected = [[1,1],[1,3],[3,1]]
    weights_expected = [
        [
            [-1.0],
        ],[
            [1/6, 2/3, 1/6],
        ],[
            [1/6, 2/3, 1/6]
        ]]
    size_expected = [1,3,3]
    idx_expected = matlab_to_python_index(np.array([[1,1],[1,2],[2,1]]))
    coeff_expected = [-1,1,1]
    knots_per_dim_expected = [
        [
            [-0.5],
            [-0.5],
        ],[
            [-0.5],
            [1.0, -0.5, -2.0]
        ],[
            [1.0, -0.5, -2.0],
            [-0.5]
        ]]
    assert is_list_math_equal(S.knots, knots_expected)
    assert is_list_math_equal(S.weights, weights_expected)
    assert is_list_math_equal(S.size, size_expected)
    assert is_list_math_equal(S.knots_per_dim, knots_per_dim_expected)
    assert is_list_math_equal(S.m, m_expected)
    assert is_list_math_equal(S.coeff, coeff_expected)
    assert is_list_math_equal(S.idx, idx_expected)

def test_create_sparse_grid_idxset():
    N = 2
    w = 5
    knots = lambda n: knots_uniform(n, -1, 1, 'nonprob')
    lev2knots = lev2knots_lin
    idxset = lambda i: np.prod(i+1)  # i 0-based so +1

    S, _ = create_sparse_grid(N, w, knots, lev2knots, idxset)
    assert len(S) == 5
    knots_expected = data = [
        # ans 1
        [
            [0, 0],
            [-0.577350269189626, 0.577350269189626]
        ],
        # ans 2
        [
            [0, 0, 0, 0, 0],
            [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]
        ],
        # ans 3
        [
            [-0.577350269189626, 0.577350269189626],
            [0, 0]
        ],
        # ans 4
        [
            [-0.577350269189626, 0.577350269189626, -0.577350269189626, 0.577350269189626],
            [-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626]
        ],
        # ans 5
        [
            [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664],
            [0, 0, 0, 0, 0]
        ]
    ]

    m_expected = [[1, 2], [1, 5], [2, 1], [2, 2], [5, 1]]
    weights_expected = data = [
        [[-2, -2]],
        [[0.473853770112378, 0.957257340998733, 1.137777777777778, 0.957257340998733, 0.473853770112378]],
        [[-2, -2]],
        [[1, 1, 1, 1]],
        [[0.473853770112378, 0.957257340998733, 1.137777777777778, 0.957257340998733, 0.473853770112378]]
    ]

    size_expected = [2,5,2,4,5]
    idx_expected = matlab_to_python_index(np.array([[1, 2], [1, 5], [2, 1], [2, 2], [5, 1]]))
    coeff_expected = [-1, 1, -1, 1, 1]
    knots_per_dim_expected = [
        [
            [0],
            [-0.577350269189626, 0.577350269189626]
        ],
        [
            [0],
            [-0.9061798459386639, -0.5384693101056832, 4.143437923020070e-17, 0.5384693101056831, 0.9061798459386639]
        ],
        [
            [-0.577350269189626, 0.577350269189626],
            [0]
        ],
        [
            [-0.577350269189626, 0.577350269189626],
            [-0.577350269189626, 0.577350269189626]
        ],
        [
            [-0.9061798459386639, -0.5384693101056832, 4.143437923020070e-17, 0.5384693101056831, 0.9061798459386639],
            [0]
        ]
    ]
    assert is_list_math_equal(S.knots, knots_expected)
    assert is_list_math_equal(S.weights, weights_expected)
    assert is_list_math_equal(S.size, size_expected)
    assert is_list_math_equal(S.knots_per_dim, knots_per_dim_expected)
    assert is_list_math_equal(S.m, m_expected)
    assert is_list_math_equal(S.coeff, coeff_expected)
    assert is_list_math_equal(S.idx, idx_expected)
