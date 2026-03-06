import pytest
import numpy as np

from sgpykit import create_sparse_grid, reduce_sparse_grid, evaluate_on_sparse_grid, knots_uniform, lev2knots_lin


@pytest.mark.parametrize(
    "N, w1, w2",
    [
        pytest.param(2,3,4, id="1"),
        pytest.param(4,1,2, id="2"),
    ]
)
def test_evaluate_on_sparse_grid(N, w1, w2):
    f = lambda x: sum(x)
    S, _ = create_sparse_grid(N, w1, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Sr = reduce_sparse_grid(S)
    T, _ = create_sparse_grid(N, w2, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    evals_non_rec, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)
    evals_rec, *_ = evaluate_on_sparse_grid(f, T,
                                            Sr=Tr,
                                            evals_old=evals_old,
                                            S_old=S,
                                            Sr_old=Sr)
    np.testing.assert_allclose(evals_non_rec, evals_rec, atol=1e-15)


@pytest.mark.parametrize(
    "N, w1, w2",
    [
        pytest.param(2,3,4, id="1"),
        pytest.param(4,1,2, id="2"),
    ]
)
def test_evaluate_on_sparse_grid_from_vector(N, w1, w2):
    f = lambda x: 2*x
    S, _ = create_sparse_grid(N, w1, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Sr = reduce_sparse_grid(S)
    T, _ = create_sparse_grid(N, w2, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)
    evals_non_rec, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Tr)
    evals_old, *_ = evaluate_on_sparse_grid(f, S=None, Sr=Sr)
    evals_rec, *_ = evaluate_on_sparse_grid(f, T,
                                            Sr=Tr,
                                            evals_old=evals_old,
                                            S_old=S,
                                            Sr_old=Sr)
    np.testing.assert_allclose(evals_non_rec, evals_rec, atol=1e-15)


def test_evaluate_on_sparse_grid_benchmark(benchmark):
    f = lambda x: sum(x)
    N = 8
    w = 6
    T, _ = create_sparse_grid(N, w, lambda n: knots_uniform(n, -1, 1), lev2knots_lin)
    Tr = reduce_sparse_grid(T)

    def func(T, Tr):
        evals_1, *_ = evaluate_on_sparse_grid(f, T, Sr=Tr,
                                              evals_old=[], S_old=[], Sr_old=[],
                                              # SGMK interface, so []s have to be provided here
                                              )
        return evals_1[0]

    result = benchmark(func, T=T, Tr=Tr)
    np.testing.assert_almost_equal(sum(result), 0.0)
