import pytest
import numpy as np

from sgpykit.tools.idxset_functions import check_set_admissibility


@pytest.mark.parametrize(
    "Ia,expected2",
    [
        pytest.param(1, [1, 1.0], id='0'),
        pytest.param([1, 1], [1, np.array([[1., 1.]])], id='1'),
        pytest.param(np.array([[1, 2], [1, 1]]), [1, np.array([[1., 1.], [1., 2.]])], id='2'),
        pytest.param(np.array([[2, 3], [2, 1]]), [0, np.array([[1., 1.], [1., 2.], [1., 3.], [2., 1.], [2., 2.], [2., 3.]])], id='3'),
        pytest.param(np.array([[1, 1],[1, 2],[1, 3],[1, 4],[2, 1],[2, 2]]), [1, np.array([[1, 1],[1, 2],[1, 3],[1, 4],[2, 1],[2, 2]])], id='4'),
    ]
)
def test_check_set_admissibility(Ia, expected2):
    I0 = np.atleast_1d(Ia)
    result1, result2 = check_set_admissibility(Ia)
    np.testing.assert_array_almost_equal(result1, expected2[0])  # 0 or 1, false or true
    np.testing.assert_array_almost_equal(result2, expected2[1])  # index set
    assert np.all(I0 == np.atleast_1d(Ia))