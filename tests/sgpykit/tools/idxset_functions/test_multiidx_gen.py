import pytest
import numpy as np

from sgpykit.tools.idxset_functions import multiidx_gen


@pytest.mark.parametrize(
    "N,rule,w,base,expected",
    [
        pytest.param(1,lambda i: sum(i-1),0,1, 1.0, id='0'),
        pytest.param(1,lambda i: sum(i-1),0,2, [], id='1'),
        # pytest.param(1,lambda i: sum(i-1),0,3, [], id='2'),
        pytest.param(1,lambda i: sum(i-1),1,1, np.array([[1.], [2.]]), id='3'),
        pytest.param(1,lambda i: sum(i-1),1,2, 2.0, id='4'),
        pytest.param(1,lambda i: sum(i-1),1,3, [], id='5'),
        pytest.param(1,lambda i: sum(i-1),4,1, np.array([[1.], [2.], [3.], [4.], [5.]]), id='6'),
        # pytest.param(1,lambda i: sum(i-1),4,2, np.array([[2.], [3.], [4.], [5.]]), id='7'),
        pytest.param(1,lambda i: sum(i-1),4,3, np.array([[3.], [4.], [5.]]), id='8'),
        pytest.param(2,lambda i: sum(i-1),0,1, np.array([[1., 1.]]), id='9'),
        pytest.param(2,lambda i: sum(i-1),0,2, [], id='10'),
        # pytest.param(2,lambda i: sum(i-1),0,3, [], id='11'),
        pytest.param(2,lambda i: sum(i-1),1,1, np.array([[1., 1.], [1., 2.], [2., 1.]]), id='12'),
        pytest.param(2,lambda i: sum(i-1),1,2, [], id='13'),
        # pytest.param(2,lambda i: sum(i-1),1,3, [], id='14'),
        pytest.param(2,lambda i: sum(i-1),4,1, np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.], [2., 1.], [2., 2.], [2., 3.], [2., 4.], [3., 1.], [3., 2.], [3., 3.], [4., 1.], [4., 2.], [5., 1.]]), id='15'),
        pytest.param(2,lambda i: sum(i-1),4,2, np.array([[2., 2.], [2., 3.], [2., 4.], [3., 2.], [3., 3.], [4., 2.]]), id='16'),
        pytest.param(2,lambda i: sum(i-1),4,3, np.array([[3., 3.]]), id='17'),
        pytest.param(3,lambda i: sum(i-1),0,1, np.array([[1., 1., 1.]]), id='18'),
        pytest.param(3,lambda i: sum(i-1),0,2, [], id='19'),
        # pytest.param(3,lambda i: sum(i-1),0,3, [], id='20'),
        pytest.param(3,lambda i: sum(i-1),1,1, np.array([[1., 1., 1.], [1., 1., 2.], [1., 2., 1.], [2., 1., 1.]]), id='21'),
        pytest.param(3,lambda i: sum(i-1),1,2, [], id='22'),
        # pytest.param(3,lambda i: sum(i-1),1,3, [], id='23'),
        pytest.param(3,lambda i: sum(i-1),4,1, np.array([[1., 1., 1.], [1., 1., 2.], [1., 1., 3.], [1., 1., 4.], [1., 1., 5.], [1., 2., 1.], [1., 2., 2.], [1., 2., 3.], [1., 2., 4.], [1., 3., 1.], [1., 3., 2.], [1., 3., 3.], [1., 4., 1.], [1., 4., 2.], [1., 5., 1.], [2., 1., 1.], [2., 1., 2.], [2., 1., 3.], [2., 1., 4.], [2., 2., 1.], [2., 2., 2.], [2., 2., 3.], [2., 3., 1.], [2., 3., 2.], [2., 4., 1.], [3., 1., 1.], [3., 1., 2.], [3., 1., 3.], [3., 2., 1.], [3., 2., 2.], [3., 3., 1.], [4., 1., 1.], [4., 1., 2.], [4., 2., 1.], [5., 1., 1.]]), id='24'),
        pytest.param(3,lambda i: sum(i-1),4,2, np.array([[2., 2., 2.], [2., 2., 3.], [2., 3., 2.], [3., 2., 2.]]), id='25'),
        pytest.param(3,lambda i: sum(i-1),4,3, [], id='26'),
    ]
)
def test_multiidx_gen(N, rule, w, base, expected):
    result = multiidx_gen(N, rule, w, base)
    np.testing.assert_array_almost_equal(result, expected)


def test_multiidx_gen_prod():
    N = 2
    w = 5
    idxset = lambda i: np.prod(i)
    C = multiidx_gen(N=N, rule=idxset, w=w, base=1)
    assert np.allclose(C, np.array([
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [2, 2],
            [3, 1],
            [4, 1],
            [5, 1]
        ], dtype=np.int64))


@pytest.mark.parametrize(
    "N,w,idxset",
    [
        pytest.param(9,8, lambda i: np.sum(i-1, axis=0), id='0'),
        pytest.param(9,8, lambda i: sum(i-1), id='1'),
        pytest.param(9,8, lambda i: np.prod(i), id='2'),
    ]
)
def test_multiidx_gen_benchmark(benchmark, N, w, idxset):
    idxset = lambda i: np.sum(i, axis=0)

    def func(N, rule, w, base):
        return multiidx_gen(N=N, rule=rule, w=w, base=base)

    benchmark(func, N=N, rule=idxset, w=w, base=1)
    pass
