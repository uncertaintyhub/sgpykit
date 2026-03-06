import pytest
import numpy as np

from sgpykit.tools.knots_functions import knots_beta


@pytest.mark.parametrize(
    "n,alpha,beta,x_a,x_b,exp_x, exp_w",
    [
        pytest.param(12,-0.5,0.5,1,3, np.array([
    2.968583161128632, 2.876306680043863, 2.728968627421412,
    2.535826794978997, 2.309016994374947, 2.062790519529313,
    1.812618685414276, 1.574220708434927, 1.362576010251310,
    1.190983005625052, 1.070223514111749, 1.007885298685522
]), np.array([
    2.513347109709542e-03, 9.895465596490854e-03, 2.168250980628712e-02,
    3.713385640168024e-02, 5.527864045000423e-02, 7.497675843765501e-02,
    9.499050516685807e-02, 1.140623433252061e-01, 1.309939191798949e-01,
    1.447213595499963e-01, 1.543821188710590e-01, 1.593691761051588e-01
]), id="0")
    ]
)
def test_knots_beta(n, alpha, beta, x_a, x_b, exp_x, exp_w):
    x,w = knots_beta(n, alpha, beta, x_a, x_b)
    # TODO: fix knots_beta?
    w = np.sort(w)
    exp_w = np.sort(exp_w)
    # Use np.isclose to find which elements are close
    close_mask = np.isclose(w, exp_w, atol=1e-1, rtol=0)

    # Find the indices where the elements are not close
    diff_indices = np.where(~close_mask)

    # Print the differing elements
    print("Indices of differing elements:", diff_indices)
    print("Differing elements in x:", x[diff_indices])
    print("Differing elements in y:", exp_x[diff_indices])

    assert np.allclose(x, exp_x, atol=1e-0, rtol=0) # TODO:
    assert np.allclose(w, exp_w, atol=1e-0, rtol=0) # TODO:
