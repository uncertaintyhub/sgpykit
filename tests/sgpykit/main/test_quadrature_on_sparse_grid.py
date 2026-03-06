import pytest
import numpy as np

from sgpykit import create_sparse_grid, reduce_sparse_grid, lev2knots_doubling, knots_CC, \
    quadrature_on_sparse_grid


def test_quadrature_on_sparse_grid():
    f = lambda x, b: np.prod(1.0 / np.sqrt(x + b), axis=0, keepdims=True)
    b = 3
    N = 4
    w = 4
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    result,_ = quadrature_on_sparse_grid(lambda x: f(x, b), S=None, Sr=Sr)
    assert np.isclose(result, 1.883984044753591)


def test_quadrature_on_sparse_grid_recycle():
    f = lambda x, b: np.prod((x + b) ** -0.5, axis=0)
    b = 5
    N = 2
    # the starting grid
    w = 3
    knots = lambda n: knots_CC(n, -2, 1, 'prob')
    S, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    IS, evals_S = quadrature_on_sparse_grid(lambda x: f(x, b), S=None, Sr=Sr)
    assert np.isclose(IS, 0.228763833671746)
    # the new grid
    w = w+1
    T, _ = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Tr = reduce_sparse_grid(T)
    IT_rec, _ = quadrature_on_sparse_grid(lambda x: f(x, b), S=T, Sr=Tr, evals_old=evals_S, S_old=S, Sr_old=Sr)
    assert np.isclose(IT_rec, 0.228763833671746)
