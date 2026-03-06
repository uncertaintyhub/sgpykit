import pytest
import numpy as np

import sgpykit as sg


def test_compute_sobol_indices_from_sparse_grid():
    import numpy as np

    # Define the domain
    aa = np.array([-1, -1, -1])
    bb = np.array([1, 1, 1])

    # Define the functions
    def f1(x):
        x = np.atleast_2d(x)
        return 1 + x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2

    def f2(x):
        x = np.atleast_2d(x)
        return 1 + 5 * x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2

    def f3(x):
        x = np.atleast_2d(x)
        return 1 / (1 + x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2)

    def f4(x):
        x = np.atleast_2d(x)
        return 1 / (1 + 5 * x[0, :] ** 2 + x[1, :] ** 2 + x[2, :] ** 2)

    def f(x):
        return np.vstack((f1(x), f2(x), f3(x), f4(x)))

    # We expect to see these results:
    #   f1: has no mixed effects, so the principal and total Sobol indices are identical. Also, it's isotropic, so the indices of each variable are identical
    #   f2: no mixed effects as f1, but y_1 contributes more to the variability of f so it has a larger Sobol total/principal index
    #   f3: this function has mixed effects (partial derivatives are nonzero),  so the principal and total Sobol index will be different, but equal among random variables
    #   f4: mixed effects, and y_1 contributes more to the variability of f so it has larger Sobol indices

    # Generate a sparse grid
    domain = np.vstack((aa, bb))
    knots = lambda n: sg.knots_CC(n, -1, 1, 'nonprob')
    N = len(aa)
    w = 5
    S, _ = sg.create_sparse_grid(N, w, knots, sg.lev2knots_doubling)
    Sr = sg.reduce_sparse_grid(S)

    values_on_grid, *_ = sg.evaluate_on_sparse_grid(f, S=None, Sr=Sr)

    # Compute Sobol indices
    Sob_i1, Tot_Sob_i1, Mean1, Var1 = sg.compute_sobol_indices_from_sparse_grid(S, Sr, values_on_grid[0, :], domain,
                                                                                'legendre')
    Sob_i2, Tot_Sob_i2, Mean2, Var2 = sg.compute_sobol_indices_from_sparse_grid(S, Sr, values_on_grid[1, :], domain,
                                                                                'legendre')
    Sob_i3, Tot_Sob_i3, Mean3, Var3 = sg.compute_sobol_indices_from_sparse_grid(S, Sr, values_on_grid[2, :], domain,
                                                                                'legendre')
    Sob_i4, Tot_Sob_i4, Mean4, Var4 = sg.compute_sobol_indices_from_sparse_grid(S, Sr, values_on_grid[3, :], domain,
                                                                                'legendre')

    assert np.allclose(Sob_i1, [0.333333333333332, 0.333333333333332, 0.333333333333334])
    assert np.allclose(Sob_i2, [9.259259259259250e-01, 3.703703703703742e-02, 3.703703703703728e-02])
    assert np.allclose(Sob_i3, [0.308069117291115, 0.308069117291115, 0.308069117291115])
    assert np.allclose(Sob_i4, [8.028047268457712e-01, 5.821388472237152e-02, 5.821388472237143e-02])

    assert np.allclose(Tot_Sob_i1, [0.333333333333332, 0.333333333333332, 0.333333333333334])
    assert np.allclose(Tot_Sob_i2, [9.259259259259250e-01, 3.703703703703742e-02, 3.703703703703728e-02])
    assert np.allclose(Tot_Sob_i3, [0.359973117419094, 0.359973117419096, 0.359973117419096])
    assert np.allclose(Tot_Sob_i4, [0.879015564722607, 0.103452021601705, 0.103452021601705])

    assert np.isclose(Mean1, 1.999999999999995)
    assert np.isclose(Var1, 0.266666666666667)