import pytest
import numpy as np

from sgpykit import lev2knots_doubling, tensor_grid, knots_CC, tensor_to_sparse
from sgpykit.util.checks import is_list_math_equal


def test_tensor_to_sparse():
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    N = 3
    ii = np.array([1, 2, 3])
    m = lev2knots_doubling(ii)
    t = tensor_grid(N, m, knots)
    s = tensor_to_sparse(t)
    assert s.coeff == 1
    assert np.allclose(s.knots, t.knots)
    assert np.allclose(s.weights, t.weights)
    assert np.allclose(s.m, t.m)
    assert is_list_math_equal(s.knots_per_dim, t.knots_per_dim)
    assert np.all(s.idx == [1,3,5])