import pytest
import numpy as np

import sgpykit as sg


@pytest.mark.parametrize(
    "ii,expected",
    [
        # Test case 1: Simple 2D case with 0-based indexing
        pytest.param(
            np.array([1, 2]),
            np.array([
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2]
            ]),
            id="simple_2d_case"
        ),

        # Test case 2: 3D case with one dimension at level 0
        pytest.param(
            np.array([1, 0, 2]),
            np.array([
                [0, 0, 1],
                [0, 0, 2],
                [1, 0, 1],
                [1, 0, 2]
            ]),
            id="3d_case_with_zero_level"
        ),

        # Test case 3: All dimensions at level 0 (should return just the zero vector)
        pytest.param(
            np.array([0, 0, 0]),
            np.array([[0, 0, 0]]),
            id="all_zeros"
        ),

        # Test case 4: Single dimension case
        pytest.param(
            np.array([2]),
            np.array([
                [1],
                [2]
            ]),
            id="single_dimension"
        ),

        # Test case 5: 3D case with all dimensions > 0
        pytest.param(
            np.array([1, 2, 3]),
            np.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 2],
                [0, 2, 3],
                [1, 1, 2],
                [1, 1, 3],
                [1, 2, 2],
                [1, 2, 3]
            ]),
            id="3d_all_positive"
        ),

        # Test case 6: Higher dimensional case (4D)
        pytest.param(
            np.array([1, 1, 0, 2]),
            np.array([
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 1, 0, 1],
                [0, 1, 0, 2],
                [1, 0, 0, 1],
                [1, 0, 0, 2],
                [1, 1, 0, 1],
                [1, 1, 0, 2]
            ]),
            id="4d_case"
        ),
    ]
)
def test_delta_to_combitec(ii, expected):
    result = sg.delta_to_combitec(ii)
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"