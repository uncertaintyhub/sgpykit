import pytest
import numpy as np

from sgpykit import knots_uniform, create_sparse_grid, reduce_sparse_grid, lege_eval_multidim, convert_to_modal
from sgpykit.tools.lev2knots_functions import lev2knots_lin


def test_convert_to_modal():
    # the sparse grid
    N = 2
    w = 5
    knots = lambda n: knots_uniform(n, -1, 1, 'nonprob')
    lev2knots = lev2knots_lin
    idxset = lambda i: np.prod(i+1)

    S, _ = create_sparse_grid(N, w, knots, lev2knots, idxset)
    Sr = reduce_sparse_grid(S)

    # the domain of the grid
    domain = np.vstack((-np.ones(N), np.ones(N)))

    # compute a legendre polynomial over the sparse grid
    X = Sr.knots
    nodal_values = 4 * lege_eval_multidim(X, [4, 0], -1, 1) + 2 * lege_eval_multidim(X, [1, 1], -1, 1)

    # conversion from the points to the legendre polynomial. I should recover it exactly
    modal_coeffs, K = convert_to_modal(S, Sr, nodal_values, domain, 'legendre')

    result = np.hstack((K, modal_coeffs.astype(np.int64)))
    expected = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 0],
        [0, 3, 0],
        [0, 4, 0],
        [1, 0, 0],
        [1, 1, 2],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 4]
    ])
    np.testing.assert_array_equal(result, expected)
