import pytest
import numpy as np

from sgpykit.src import find_lexicographic


# Test cases
@pytest.mark.parametrize(
    "lookfor, I, nocheck, expected_found, expected_pos",
    [
        # Basic case: lookfor is in the matrix
        ([1, 2, 3], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, True, 0),
        ([4, 5, 6], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, True, 1),
        ([7, 8, 9], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, True, 2),

        # Edge case: lookfor is not in the matrix
        ([10, 11, 12], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, False, None),

        # Edge case: matrix is empty
        ([1, 2, 3], np.array([]), None, False, None),

        # Edge case: matrix has only one row
        ([1, 2, 3], np.array([[1, 2, 3]]), None, True, 0),
        ([4, 5, 6], np.array([[1, 2, 3]]), None, False, None),

        # Special case: matrix is not sorted and nocheck is used
        ([4, 5, 6], np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]]), 'nocheck', True, 1),

        # Special case: matrix is not sorted and nocheck is not used (should raise an exception)
        ([4, 5, 6], np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]]), None, None, None),

        # Special case: invalid nocheck parameter (should raise an exception)
        ([4, 5, 6], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'invalid', None, None),
    ]
)
def test_find_lexicographic(lookfor, I, nocheck, expected_found, expected_pos):
    if expected_found is None:
        with pytest.raises(Exception):
            find_lexicographic(lookfor, I, nocheck)
    else:
        found, pos, iter = find_lexicographic(lookfor, I, nocheck)
        assert found == expected_found
        assert pos == expected_pos