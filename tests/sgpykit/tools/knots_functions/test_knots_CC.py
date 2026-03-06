import pytest
import numpy as np

from sgpykit.tools.knots_functions.knots_CC import knots_CC


@pytest.mark.parametrize(
    "nn,x_a,x_b,whichrho,expected",
    [
        pytest.param(1, -1, 1, 'nonprob', [0,2], id=""),
        pytest.param(3, -1, 1, 'nonprob', [[1, 6.1232e-17, -1],
                                           [0.333333333333333, 1.333333333333333, 0.333333333333333]], id="")
    ]
)
def test_knots_CC(nn, x_a, x_b, whichrho, expected):
    values = knots_CC(nn, x_a, x_b, whichrho)
    assert np.allclose(values, expected)


