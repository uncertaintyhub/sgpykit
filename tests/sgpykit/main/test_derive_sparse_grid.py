import pytest
import numpy as np

import sgpykit as sg


def test_derive_sparse_grid():
    f = lambda x: 1.0 / (1.0 + 0.5 * np.sum(x ** 2, axis=0))

    N = 2
    aa = np.array([4, 1])
    bb = np.array([6, 5])
    domain = np.vstack((aa, bb))

    w = 4
    knots1 = lambda n: sg.knots_CC(n, aa[0], bb[0])
    knots2 = lambda n: sg.knots_CC(n, aa[1], bb[1])
    S,_ = sg.create_sparse_grid(N, w, [knots1, knots2], sg.lev2knots_doubling)
    Sr = sg.reduce_sparse_grid(S)
    f_on_grid,*_ = sg.evaluate_on_sparse_grid(f, S=None, Sr=Sr)

    x1 = np.linspace(aa[0], bb[0], 10)
    x2 = np.linspace(aa[1], bb[1], 30)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    eval_points = np.vstack((X1.ravel(), X2.ravel()))

    grad = sg.derive_sparse_grid(S, Sr, f_on_grid, domain, eval_points)
    grad_expected_stats = [ # min, median, max
        [-4.432133275493465e-02, -1.458587962658783e-02, -6.046862596753576e-03],
        [-1.700072112670314e-02, -8.179940151013115e-03, -2.629684020712120e-03]
    ]
    grad_stats = np.array([
        np.min(grad, axis=1), np.median(grad, axis=1), np.max(grad, axis=1)
    ]).T
    np.testing.assert_array_almost_equal(grad_stats, grad_expected_stats)