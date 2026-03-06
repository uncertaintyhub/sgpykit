import pytest
import numpy as np

from sgpykit.tools.idxset_functions import check_index_admissibility
from sgpykit.util.misc import matlab_to_python_index


@pytest.mark.parametrize(
    "idx,idx_set,sort_option,expected3",
    [
        pytest.param([1, 1],[1, 1],'sorting', [1, np.array([1., 1.]), []], id='0'),
        pytest.param([1, 1],[1, 1],None, [1, np.array([1., 1.]), []], id='1'),
        pytest.param([1, 1],np.array([[1, 1], [1, 2]]),'sorting', [1, np.array([[1., 1.], [1., 2.]]), []], id='2'),
        pytest.param([1, 1],np.array([[1, 1], [1, 2]]),None, [1, np.array([[1., 1.], [1., 2.]]), []], id='3'),
        pytest.param([1, 1],np.array([[2, 3], [2, 1]]),'sorting', [1, np.array([[2., 1.], [2., 3.]]), []], id='4'),
        pytest.param([1, 1],np.array([[2, 3], [2, 1]]),None, [1, np.array([[2., 3.], [2., 1.]]), []], id='5'),
        pytest.param(np.array([[1, 2], [1, 1]]),[1, 1],'sorting', [1, np.array([1., 1.]), []], id='6'),
        pytest.param(np.array([[1, 2], [1, 1]]),[1, 1],None, [1, np.array([1., 1.]), []], id='7'),
        pytest.param(np.array([[1, 2], [1, 1]]),np.array([[1, 1], [1, 2]]),'sorting', [1, np.array([[1., 1.], [1., 2.]]), []], id='8'),
        pytest.param(np.array([[1, 2], [1, 1]]),np.array([[1, 1], [1, 2]]),None, [1, np.array([[1., 1.], [1., 2.]]), []], id='9'),
        pytest.param(np.array([[1, 2], [1, 1]]),np.array([[2, 3], [2, 1]]),'sorting', [0, np.array([[1., 1.], [2., 1.], [2., 3.]]), np.array([[1., 1.]])], id='10'),
        pytest.param(np.array([[1, 2], [1, 1]]),np.array([[2, 3], [2, 1]]),None, [0, np.array([[2., 3.], [2., 1.], [1., 1.]]), np.array([[1., 1.]])], id='11'),
        pytest.param(np.array([[2, 3], [2, 1]]),[1, 1],'sorting', [0, np.array([[1., 1.], [1., 2.], [1., 3.], [2., 1.], [2., 2.]]), np.array([[1., 2.], [1., 3.], [2., 1.], [2., 2.]])], id='12'),
        pytest.param(np.array([[2, 3], [2, 1]]),[1, 1],None, [0, np.array([[1., 1.], [1., 3.], [2., 2.], [1., 2.], [2., 1.]]), np.array([[1., 3.], [2., 2.], [1., 2.], [2., 1.]])], id='13'),
        pytest.param(np.array([[2, 3], [2, 1]]),np.array([[1, 1], [1, 2]]),'sorting', [0, np.array([[1., 1.], [1., 2.], [1., 3.], [2., 1.], [2., 2.]]), np.array([[1., 3.], [2., 1.], [2., 2.]])], id='14'),
        pytest.param(np.array([[2, 3], [2, 1]]),np.array([[1, 1], [1, 2]]),None, [0, np.array([[1., 1.], [1., 2.], [1., 3.], [2., 2.], [2., 1.]]), np.array([[1., 3.], [2., 2.], [2., 1.]])], id='15'),
        pytest.param(np.array([[2, 3], [2, 1]]),np.array([[2, 3], [2, 1]]),'sorting', [0, np.array([[1., 1.], [1., 2.], [1., 3.], [2., 1.], [2., 2.], [2., 3.]]), np.array([[1., 1.], [1., 2.], [1., 3.], [2., 2.]])], id='16'),
        pytest.param(np.array([[2, 3], [2, 1]]),np.array([[2, 3], [2, 1]]),None, [0, np.array([[2., 3.], [2., 1.], [1., 3.], [2., 2.], [1., 1.], [1., 2.]]), np.array([[1., 3.], [2., 2.], [1., 1.], [1., 2.]])], id='17'),
        pytest.param(np.array([[2, 1], [1, 2]]),[1, 1],None, [1, np.array([1., 1.]), []], id='18'),
        # pytest.param(, , , id="")
    ]
)
def test_check_index_admissibility(idx, idx_set, sort_option, expected3):
    idx = matlab_to_python_index(idx)
    idx_set = matlab_to_python_index(idx_set)
    idx0 = np.atleast_1d(idx)
    idx_set0 = np.atleast_1d(idx_set)
    result1, result2, result3 = check_index_admissibility(idx, idx_set, sort_option)
    np.testing.assert_array_almost_equal(result1, expected3[0])  # 0 or 1, false or true
    np.testing.assert_array_almost_equal(result2, matlab_to_python_index(expected3[1]))  # index set
    np.testing.assert_array_almost_equal(result3, matlab_to_python_index(expected3[2]))  # index set
    assert np.all(idx0 == np.atleast_1d(idx))
    assert np.all(idx_set0 == np.atleast_1d(idx_set))
