import pytest
import numpy as np

from sgpykit.tools.rescaling_functions.get_interval_map import get_interval_map


@pytest.mark.parametrize(
    "a,b,type_,lambda_0,expected",
    [
        # TODO: fill in test data for test_get_interval_map
        pytest.param(np.array([7, -2]),np.array([1, 0]),'uniform',np.array([[1, 0], [0, 1]]), np.array([[ 1.,4.], [-1.,0.]]), id='0'),
        pytest.param(np.array([7, -2]),np.array([1, 0]),'uniform',np.array([[-5], [1]]), np.array([[19.], [ 0.]]), id='1'),
        pytest.param(np.array([7, -2]),np.array([1, 0]),'gaussian',np.array([[1, 0], [0, 1]]), np.array([[ 8.,7.], [-2., -2.]]), id='2'),
        pytest.param(np.array([7, -2]),np.array([1, 0]),'gaussian',np.array([[-5], [1]]), np.array([[ 2.], [-2.]]), id='3'),
        # pytest.param(, , , id="")
    ]
)
def test_get_interval_map(a, b, type_, lambda_0, expected):
    a0 = np.atleast_1d(a)
    b0 = np.atleast_1d(b)
    result_lambda = get_interval_map(a, b, type_)
    result = result_lambda(lambda_0)
    np.testing.assert_array_almost_equal(result, expected)
    assert np.all(a0 == np.atleast_1d(a))
    assert np.all(b0 == np.atleast_1d(b))