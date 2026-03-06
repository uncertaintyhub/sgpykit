import pytest
import numpy as np

from sgpykit.main import create_sparse_grid, reduce_sparse_grid
from sgpykit.src import compare_sparse_grids
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.util.misc import matlab_to_python_index


def test_compare_sparse_grids():
    N=2
    w=3
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S,_ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    S_old,_ = create_sparse_grid(N, w-1, knots, lev2knots_doubling) # TODO: if w2 > w1 -> error
    Sr_old = reduce_sparse_grid(S_old)
    tol = 1e-4
    pts_in_s_only, pts_in_both_grids_s, pts_in_both_grids_s_old, pts_in_s_old_only \
        = compare_sparse_grids(S, Sr, S_old, Sr_old, tol)
    np.testing.assert_array_equal(pts_in_s_only, matlab_to_python_index([2,4,6,7,9,10,12,14,16,18,20,21,23,24,26,28]))
    np.testing.assert_array_equal(pts_in_both_grids_s, matlab_to_python_index([1,3,5,8,11,13,15,17,19,22,25,27,29]))
    np.testing.assert_array_equal(pts_in_both_grids_s_old, matlab_to_python_index([1,2,3,4,5,6,7,8,9,10,11,12,13]))
    assert len(pts_in_s_old_only) == 0