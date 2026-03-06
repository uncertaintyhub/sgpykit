import pytest
import numpy as np

from sgpykit.tools.lev2knots_functions import lev2knots_doubling


@pytest.mark.parametrize(
    "i,expected",
    [
        pytest.param(0, 0, id='0'),
        pytest.param(2, 3, id='1'),
        pytest.param(10, 513, id='2'),
        pytest.param(np.array([7, 2, 0]), np.array([65.,3.,0.]), id='3'),
        pytest.param(np.array([[1], [2]]), np.array([[1.], [3.]]), id='4'),
        # pytest.param(, id="")
    ]
)
def test_lev2knots_doubling(i, expected):
    i0 = np.atleast_1d(i)
    result = lev2knots_doubling(i)
    np.testing.assert_array_almost_equal(result, expected)
    assert np.all(i0 == np.atleast_1d(i))