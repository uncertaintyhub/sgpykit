import pytest
import numpy as np
import sgpykit as sg
from sgpykit.util.checks import is_list_math_equal


def test_create_sparse_grid_multiidx_set():
    N = 2
    a, b = -1.0, 1.0
    knots = lambda n: sg.knots_CC(n, a, b)
    lev2knots = sg.lev2knots_doubling
    G = np.zeros((1, N), dtype=np.int64)
    S,_ = sg.create_sparse_grid_multiidx_set(G, knots, lev2knots)
    # expected
    knots_expected = [
        [
            [0],
            [0],
        ]]
    m_expected = [[1, 1]]
    weights_expected = [
        [
            [1],
        ]]
    size_expected = [1]
    idx_expected = [[0, 0]]
    coeff_expected = [1]
    knots_per_dim_expected = [
        [
            [0],
            [0],
        ]]
    assert is_list_math_equal(S.knots, knots_expected)
    assert is_list_math_equal(S.weights, weights_expected)
    assert is_list_math_equal(S.size, size_expected)
    assert is_list_math_equal(S.knots_per_dim, knots_per_dim_expected)
    assert is_list_math_equal(S.m, m_expected)
    assert is_list_math_equal(S.coeff, coeff_expected)
    assert is_list_math_equal(S.idx, idx_expected)


def test_create_sparse_grid_multiidx_set_recycle():
    knots = lambda n: sg.knots_normal(n, 0, 1)
    lev2knots = sg.lev2knots_lin
    ibox = [3, 4, 2, 4, 2]
    _, C = sg.multiidx_box_set(ibox, 0)
    D, _ = sg.matlab.sortrows(np.vstack((C, [1, 4, 1, 1, 5])))
    S, _ = sg.create_sparse_grid_multiidx_set(C, knots, lev2knots)
    T, _ = sg.create_sparse_grid_multiidx_set(D, knots, lev2knots)
    T_rec, _ = sg.create_sparse_grid_multiidx_set(D, knots, lev2knots, S)
    assert T.isequal_to(T_rec), "Results Mismatch"
