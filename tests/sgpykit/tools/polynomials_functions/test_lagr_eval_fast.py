import pytest
import numpy as np

from sgpykit.tools.polynomials_functions.lagr_eval_fast import lagr_eval_fast


def test_lagr_eval_fast():
    current_knot = 0.5
    other_knots = np.array([0.1, 0.2, 0.3])
    ok_len = len(other_knots)
    non_grid_points = np.array([0.4, 0.6, 0.7])
    ng_size = len(non_grid_points)

    exp = np.array([[0.25, 2.5,  5.  ],
                    [0.25, 2.5,  5.  ],
                    [0.25, 2.5,  5.  ]])
    L = lagr_eval_fast(current_knot, other_knots, ok_len, non_grid_points, ng_size)
    np.testing.assert_array_almost_equal(exp, L)


def test_lagr_eval_fast_transposed():
    current_knot = 0.5
    other_knots = np.array([0.1, 0.2, 0.3])
    ok_len = len(other_knots)
    non_grid_points = np.array([0.4, 0.6, 0.7])
    non_grid_points = np.atleast_2d(non_grid_points).T
    ng_size = len(non_grid_points)

    exp = np.array([[0.25, 0.25,  0.25  ],
                    [2.5, 2.5,  2.5  ],
                    [5., 5.,  5.  ]])
    L = lagr_eval_fast(current_knot, other_knots, ok_len, non_grid_points, ng_size)
    np.testing.assert_array_almost_equal(exp, L)


@pytest.mark.parametrize("n_knots, n_points", [
    (10, 100),          # Small case
    (50, 1000),         # Medium case
    (100, 10000),       # Large case
    (200, 100000),      # Very large case
])
def test_lagr_eval_fast_benchmark(benchmark, n_knots, n_points):
    # Setup test data
    current_knot = 0.5
    other_knots = np.linspace(0, 1, n_knots, endpoint=False)
    other_knots = other_knots[other_knots != current_knot]  # Ensure current_knot not in other_knots
    ok_len = len(other_knots)
    non_grid_points = np.linspace(0, 1, n_points)
    ng_size = non_grid_points.shape

    # Benchmark the function
    result = benchmark(
        lagr_eval_fast,
        current_knot,
        other_knots,
        ok_len,
        non_grid_points,
        ng_size
    )

    # Basic verification that the result has the correct shape
    assert result.shape == ng_size

    # Verify that L(current_knot) = 1 exactly when current_knot is in non_grid_points
    if current_knot in non_grid_points:
        idx = np.where(non_grid_points == current_knot)[0][0]
        assert np.isclose(result[idx], 1.0, rtol=1e-10)

    # Verify that L(other_knots) = 0 exactly when other_knots are in non_grid_points
    for knot in other_knots:
        if knot in non_grid_points:
            idx = np.where(non_grid_points == knot)[0][0]
            assert np.isclose(result[idx], 0.0, atol=1e-10)
