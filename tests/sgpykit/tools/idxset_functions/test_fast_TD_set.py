import pytest
import numpy as np

from sgpykit.tools.idxset_functions import fast_TD_set
from sgpykit.util.misc import matlab_to_python_index


@pytest.mark.parametrize(
    "N,w,expected",
    [
        pytest.param(1,0, 1.0, id='0'),
        pytest.param(1,1, np.array([[1.], [2.]]), id='1'),
        pytest.param(1,4, np.array([[1.], [2.], [3.], [4.], [5.]]), id='2'),
        pytest.param(2,0, np.array([[1., 1.]]), id='3'),
        pytest.param(2,1, np.array([[1., 1.], [1., 2.], [2., 1.]]), id='4'),
        pytest.param(2,4, np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.], [1., 5.], [2., 1.], [2., 2.], [2., 3.], [2., 4.], [3., 1.], [3., 2.], [3., 3.], [4., 1.], [4., 2.], [5., 1.]]), id='5'),
        pytest.param(3,0, np.array([[1., 1., 1.]]), id='6'),
        pytest.param(3,1, np.array([[1., 1., 1.], [1., 1., 2.], [1., 2., 1.], [2., 1., 1.]]), id='7'),
        pytest.param(3,4, np.array([[1., 1., 1.], [1., 1., 2.], [1., 1., 3.], [1., 1., 4.], [1., 1., 5.], [1., 2., 1.], [1., 2., 2.], [1., 2., 3.], [1., 2., 4.], [1., 3., 1.], [1., 3., 2.], [1., 3., 3.], [1., 4., 1.], [1., 4., 2.], [1., 5., 1.], [2., 1., 1.], [2., 1., 2.], [2., 1., 3.], [2., 1., 4.], [2., 2., 1.], [2., 2., 2.], [2., 2., 3.], [2., 3., 1.], [2., 3., 2.], [2., 4., 1.], [3., 1., 1.], [3., 1., 2.], [3., 1., 3.], [3., 2., 1.], [3., 2., 2.], [3., 3., 1.], [4., 1., 1.], [4., 1., 2.], [4., 2., 1.], [5., 1., 1.]]), id='8'),
    ]
)
def test_fast_TD_set(N, w, expected):
    result = fast_TD_set(N, w)
    np.testing.assert_array_almost_equal(result, matlab_to_python_index(expected))



@pytest.mark.parametrize(
    "N,w,expected",
    [
        pytest.param(3,1, 15, id='0'),
        pytest.param(4,6, 1848, id='1'),
        pytest.param(6,4, 1980, id='2'),
        pytest.param(7,5, 9009, id='3'),
        pytest.param(8,6, 40040, id='4'),
    ]
)
def test_fast_TD_set_big(N, w, expected):
    result = fast_TD_set(N, w, base=1)
    assert np.sum(result)==expected


def test_fast_TD_set_benchmark(benchmark):
    N = 8
    w = 6

    def func(N, w, base):
        return fast_TD_set(N, w, base)

    result = benchmark(func, N=N, w=w, base=1)
    np.testing.assert_almost_equal(np.sum(result), 40040)