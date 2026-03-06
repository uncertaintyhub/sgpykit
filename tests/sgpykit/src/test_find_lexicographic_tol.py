import pytest
import numpy as np

from sgpykit.src import find_lexicographic_tol


@pytest.mark.parametrize(
    "lookfor, I, tol, expected_bool, expected_pos",
    [
        # Empty matrix
        ([1, 2], np.array([]), 1e-14, False, []),
        # Found at the beginning
        ([1, 2], np.array([[1, 2], [3, 4], [5, 6]]), 1e-14, True, 0),
        # Found in the middle
        ([3, 4], np.array([[1, 2], [3, 4], [5, 6]]), 1e-14, True, 1),
        # Found at the end
        ([5, 6], np.array([[1, 2], [3, 4], [5, 6]]), 1e-14, True, 2),
        # Not found
        ([2, 3], np.array([[1, 2], [3, 4], [5, 6]]), 1e-14, False, []),
        # Found with noise
        ([1, 2], np.array([[1 + 1e-13, 2 + 1e-13], [3, 4], [5, 6]]), 1e-12, True, 0),
        # Not found with noise
        ([1, 2], np.array([[1 + 1e-13, 2 + 1e-13], [3, 4], [5, 6]]), 1e-14, False, []),
    ],
)
def test_find_lexicographic_tol(lookfor, I, tol, expected_bool, expected_pos):
    retbool, pos, _ = find_lexicographic_tol(lookfor, I, tol)
    assert expected_bool == retbool, f"Expected {expected_bool}, but got {retbool}"
    assert expected_pos == pos, f"Expected {expected_pos}, but got {pos}"