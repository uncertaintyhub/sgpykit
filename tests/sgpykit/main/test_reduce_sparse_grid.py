import pytest
import numpy as np


from sgpykit import tensor_grid, tensor_to_sparse
from sgpykit.main import create_sparse_grid
from sgpykit.main import reduce_sparse_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.util.checks import is_list_math_equal
from sgpykit.util.misc import matlab_to_python_index, python_to_matlab_index


def test_reduce_sparse_grid_small():
    N = 2  # do not change
    w = 1  # do not change
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)

    # expected
    knots_expected = np.array(
        [[-1.0, 0, 0, 0, 1.0],
         [0, -1.0, 0, 1.0, 0]])
    m_expected = matlab_to_python_index([7, 4, 3, 2, 5])
    weights_expected = np.array([2 / 3, 2 / 3, 4 / 3, 2 / 3, 2 / 3])
    n_expected = matlab_to_python_index([3, 4, 3, 2, 5, 3, 1])

    tol = 1e-4  # TODO: check
    assert np.allclose(Sr.knots, knots_expected, atol=tol, rtol=0)
    assert len(Sr.m) == len(m_expected)
    assert np.all(Sr.m == m_expected)
    assert np.all(Sr.n == n_expected)
    assert np.allclose(Sr.weights, weights_expected, atol=tol, rtol=0)


def test_reduce_sparse_grid():
    N = 2 # do not change
    w = 2 # do not change
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    # expected
    knots_expected = np.array([[-1.0000,-1.0000,-1.0000,-0.7071,0.0000,0,0.0000,0, 0.0000, 0.7071, 1.0000, 1.0000, 1.0000],
      [-1.0000, 0.0000,1.0000,0,  -1.0000,  -0.7071,   0.0000,   0.7071, 1.0000,0,-1.0000,0.0000,1.0000]])
    m_expected = matlab_to_python_index([20, 17, 14, 24, 19,  7, 16,  5, 13, 22, 18, 15, 12])
    weights_expected = np.array([0.111111, -0.088889, 0.111111, 1.066667,  -0.088889, 1.066667,  -0.355556, 1.066667,  -0.088889, 1.066667, 0.111111, -0.088889, 0.111111])
    n_expected = matlab_to_python_index([9,  7,  5,  9,  8,  7,  6,  5, 12,  7,  2, 13,  9,  3, 12, 7,  2, 11,  5,  1, 12, 10,  7,  4,  2])

    tol=1e-4
    assert np.allclose(Sr.knots, knots_expected, atol=tol, rtol=0)
    assert np.all(Sr.m == m_expected)
    assert np.all(Sr.n == n_expected)
    assert np.allclose(Sr.weights, weights_expected, atol=tol, rtol=0)


def test_reduce_sparse_grid_from_tensorgrid():
    N = 2
    # let's build a tensor grid with the following choices
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    ii = np.array([3, 2])
    m = lev2knots_doubling(ii)
    T = tensor_grid(N, m, knots)
    S = tensor_to_sparse(T)
    assert S.size == 15
    assert np.allclose(S.weights, np.array([
        2.222222222222224e-02, 1.777777777777778e-01, 2.666666666666667e-01,
        1.777777777777778e-01, 2.222222222222224e-02, 8.888888888888895e-02,
        7.111111111111111e-01, 1.066666666666667e+00, 7.111111111111111e-01,
        8.888888888888895e-02, 2.222222222222224e-02, 1.777777777777778e-01,
        2.666666666666667e-01, 1.777777777777778e-01, 2.222222222222224e-02
    ]))
    assert np.all(S.m == [5, 3])
    assert is_list_math_equal(S.knots_per_dim, [[1,7.071067811865476e-01,0,-7.071067811865475e-01,-1], [1,0,-1]])
    assert np.all(S.idx == [5,3])
    #
    Sr = reduce_sparse_grid(S)
    #
    assert Sr.size == 15
    assert np.all(Sr.m == np.array([14, 9, 4, 13, 8, 3, 12, 7, 2, 11, 6, 1, 10, 5, 0]))
    assert np.all(Sr.n == np.array([14, 11, 8, 5, 2, 13, 10, 7, 4, 1, 12, 9, 6, 3, 0]))
    assert np.allclose(Sr.weights, np.array([
        2.222222222222224e-02, 8.888888888888895e-02, 2.222222222222224e-02,
        1.777777777777778e-01, 7.111111111111111e-01, 1.777777777777778e-01,
        2.666666666666667e-01, 1.066666666666667e+00, 2.666666666666667e-01,
        1.777777777777778e-01, 7.111111111111111e-01, 1.777777777777778e-01,
        2.222222222222224e-02, 8.888888888888895e-02, 2.222222222222224e-02
    ]))


def test_reduce_sparse_grid_shift_prob():
    N = 2  # do not change
    w = 1  # do not change
    knots = lambda n: knots_CC(n, -2, 1, 'prob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)

    # expected
    knots_expected = [
        [-2.0, -0.5, -0.5, -0.5, 1.0],
        [-0.5, -2.0, -0.5, 1.0, -0.5]
    ]
    m_expected = matlab_to_python_index([7,4,3,2,5])
    weights_expected = [1/6, 1/6, 1/3, 1/6, 1/6]
    size_expected = 5
    n_expected = matlab_to_python_index([3,4,3,2,5,3,1])
    assert is_list_math_equal(Sr.knots, knots_expected)
    assert is_list_math_equal(Sr.weights, weights_expected)
    assert Sr.size == size_expected
    assert is_list_math_equal(Sr.n, n_expected)
    assert is_list_math_equal(Sr.m, m_expected)