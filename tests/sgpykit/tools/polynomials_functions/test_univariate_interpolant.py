import pytest
import numpy as np

from sgpykit import univariate_interpolant


def test_univariate_interpolant():
    # Define functions
    f1 = lambda x: x ** 3
    f2 = lambda x: 2 * x

    # Interpolation points and values
    x_interp = np.linspace(-1, 2, 4)  # same as linspace(-1,2,4) in MATLAB
    F_interp = np.vstack([f1(x_interp), f2(x_interp)])  # stack row-wise

    # Evaluation grid
    x_eval = np.arange(-1, 2.0001, 0.25)  # MATLAB -1:0.25:2 includes the endpoint
    F_eval_interp = univariate_interpolant(x_interp, F_interp, x_eval)

    expected = np.array([
        [-1.000000000000000, -0.421875000000000, -0.125000000000000,
         -0.015625000000000,  0.000000000000000,  0.015625000000000,
          0.125000000000000,  0.421875000000000,  1.000000000000000,
          1.953125000000000,  3.375000000000000,  5.359375000000000,
          8.000000000000000],
        [-2.000000000000000, -1.500000000000000, -1.000000000000000,
         -0.500000000000000,  0.000000000000000,  0.500000000000000,
          1.000000000000000,  1.500000000000000,  2.000000000000000,
          2.500000000000000,  3.000000000000000,  3.500000000000000,
          4.000000000000000]
    ])

    assert np.allclose(F_eval_interp, expected)