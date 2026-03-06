import pytest
import numpy as np

import sgpykit as sg


def test_combination_technique():
    N = 2
    w = 3
    knots = lambda n: sg.knots_CC(n, -1, 1)
    lev2knots, rule = sg.define_functions_for_rule('SM', N)
    S_smolyak, I_smolyak = sg.create_sparse_grid(N, w, knots, lev2knots, rule)
    coeff_smolyak = sg.combination_technique(I_smolyak)
    coeff_expected = np.array([0, 0,-1, 1, 0,-1, 1,-1, 1, 1])
    np.testing.assert_array_almost_equal(coeff_smolyak, coeff_expected)