import pytest
import numpy as np

from sgpykit.tools.type_and_property_check_functions import islexico


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param([1, 2, 3],[1, 2, 5], True, id="0"),
        pytest.param([1, 2, 3],[1, 2, 3], True, id="1"),
        pytest.param([1, 2, 3],[1, 2, 1], False, id="2"),
        pytest.param([2, 2, 3],[1, 7, 1], False, id="3"),
        pytest.param([1, 7, 3],[1, 5, 3], False, id="4"),
    ]
)
def test_islexico(a, b, expected):
    a = np.array(a)
    b = np.array(b)
    retval = islexico(a, b)
    assert retval == expected