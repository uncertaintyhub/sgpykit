import pytest
import numpy as np

from sgpykit.tools.type_and_property_check_functions import islexico_tol


@pytest.mark.parametrize(
    "a,b,tol,expected",
    [
        pytest.param([1,3],[1,4], 1e-12, True, id="0"),
        pytest.param([1+1e-13,3],[1,4], 1e-12, True, id="1"),
        pytest.param([1+1e-13,3],[1,4], 1e-14, False, id="2"),
        pytest.param([1,3+1e-11], [1+1e-11, 3], 1e-12, True, id="3"),
        pytest.param([1,4], [1, 4], 1e-12, True, id="4"),
        pytest.param([1,3,5], [1,4,2], 1e-12, True, id="5"),
        pytest.param([1,5,3], [1,2,4], 1e-12, False, id="6"),
    ]
)
def test_islexico_tol(a, b, tol, expected):
    a = np.array(a)
    b = np.array(b)
    retval = islexico_tol(a, b, tol)
    assert retval == expected