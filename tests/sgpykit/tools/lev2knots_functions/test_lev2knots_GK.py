import pytest
import numpy as np

from sgpykit.tools.lev2knots_functions import lev2knots_GK


@pytest.mark.parametrize(
    "I,expected",
    [
        pytest.param(1, 1.0, id='0'),
        pytest.param([0, 2, 1, 4], np.array([ 0.,3.,1., 19.]), id='1'),
        pytest.param([3, 5], np.array([ 9., 35.]), id='2'),
        pytest.param(np.array([5, 6, 7]), np.array([35., -1, -1]), id='3'),
        pytest.param(np.array([[1], [2]]), np.array([[1.], [3.]]), id='4'),
        # TODO: more test values like negative values and wrong datatypes? For now we rely on asserts
    ]
)
def test_lev2knots_GK(I, expected):
    I0 = np.atleast_1d(I)
    result = lev2knots_GK(I)
    np.testing.assert_array_almost_equal(result, expected)
    assert np.all(I0 == np.atleast_1d(I))

