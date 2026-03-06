import pytest
import numpy as np

import sgpykit
import sgpykit.tools.knots_functions.knots_GK
import sgpykit as sg

@pytest.mark.parametrize(
    "n, mi, sigma, expected_x, expected_w",
    [
        pytest.param(1,1,1, 1, 1, id="1"),
        pytest.param(3, 0, 1, [-1.732050807568877,0,1.732050807568877], [0.166666666666667,0.666666666666667,0.166666666666667], id="2"),
        pytest.param(9, 0, 1, [-4.184956017672732, -2.861279576057058, -1.732050807568877,
              -0.741095349994541, 0.0, 0.741095349994541,
              1.732050807568877, 2.861279576057058, 4.184956017672732], [9.426945755651738e-05, 7.996325470893528e-03, 9.485094850948504e-02,
              2.700743295779378e-01, 2.539682539682542e-01, 2.700743295779378e-01,
              9.485094850948504e-02, 7.996325470893528e-03, 9.426945755651738e-05], id="3")
    ]
)
def test_knots_GK(n, mi, sigma, expected_x, expected_w):
    x,w = sg.knots_GK(n, mi, sigma)
    assert np.allclose(x, expected_x)
    assert np.allclose(w, expected_w)
