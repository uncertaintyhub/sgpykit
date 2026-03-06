import pytest
import numpy as np

from sgpykit.util.cell import Cell
from sgpykit.util import matlab
from sgpykit.util.matlab import struct, ce, cell
from sgpykit.util.struct import Struct


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(np.array([]), True, id="0"),
        pytest.param(np.array([[1, 2, 3]]), True, id="1"),
        pytest.param(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), True, id="2"),
        pytest.param(np.array([[1, 2, 3], [3, 2, 1], [7, 8, 9]]), True, id="3"),
        pytest.param(np.array([[1, 1],[1, 2],[1, 3],[1, 4],[2, 1],[2, 2]]), True, id="4"),
        pytest.param(np.array([[1, 1],[1, 2],[1, 4],[1, 3]]), False, id="5"),
    ]
)
def test_issorted(var, expected):
    assert matlab.issorted(var, typ='rows') == expected


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1]), id="1D array"),
        pytest.param(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]]),
                     id="2D array"),
        pytest.param(np.array([1]), np.array([1]), id="1D single element"),
        pytest.param(np.array([[1]]), np.array([[1]]), id="2D single element"),
        pytest.param(np.array([]), np.array([]), id="1D empty array"),
        pytest.param(np.array([[]]), np.array([[]]), id="2D empty array"),
        pytest.param(np.array([[1, 2], [3, 4], [5, 6]]), np.array([[2, 1], [4, 3], [6, 5]]),
                     id="2D array with multiple rows"),
        pytest.param(np.array([[1, 2, 3, 4]]), np.array([[4, 3, 2, 1]]), id="2D single row"),
        pytest.param(np.array([[1], [2], [3]]), np.array([[1], [2], [3]]), id="2D single column"),
    ]
)
def test_fliplr(var, expected):
    result = matlab.fliplr(var)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(np.array([1, -3]), np.array([-1+0j, 2+0j]), id="0"),
        pytest.param(np.array([[1, -3]]), np.array([-1+0j, 2+0j]), id="1"),
        pytest.param(np.array([[1], [-3]]), np.array([-1+0j, 2+0j]), id="2"),
        pytest.param(np.array([[1, 2], [-3, 4]]), np.array([[-1+0j, 3+0j], [2+0j, -1+0j]]), id="3")
    ]
)
def test_ifft(var, expected):
    assert np.allclose(matlab.ifft(var), expected)


@pytest.mark.parametrize(
    "var,expected_x,expected_W",
    [
        pytest.param(np.array([[2, 4], [-9, 2]]), np.array([2+6j,2-6j]), np.array([[0 - 0.554700196225229j, 0 + 0.554700196225229j],[0.832050294337844 + 0j, 0.832050294337844 - 0j]]), id="0")
    ]
)
def test_eig(var, expected_x, expected_W):
    W, x = matlab.eig(var)
    assert np.allclose(x, expected_x)
    assert np.allclose(W, expected_W)


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(np.array([1, 2, 3, 4, 5]), True, id="0"),
        pytest.param(np.array([1.3, 3.3, -5.2]), True, id="1"),
        pytest.param(np.array([1+5j, -1-4j]), True, id="2"),
        pytest.param("foobar", False, id="3"),
        pytest.param(np.array([True, False, True]), False, id="4"),
        pytest.param(np.array([1.0, np.nan, np.inf]), True, id="5"),
        pytest.param(np.pi, True, id="6"),
        pytest.param([1.3, 3.3, -5.2], True, id="7")
    ]
)
def test_isnumeric(var, expected):
    assert matlab.isnumeric(var) == expected


def test_fieldnames():
    s1 = struct("field1", 0, "field2", 1, "field3", 2)
    assert set(matlab.fieldnames(s1)) == {"field1", "field2", "field3"}


def test_ce():
    c1 = ce('a', 'b', 'c')
    assert len(c1) == 3
    assert c1[0,0] == 'a' and c1[0,1] == 'b' and c1[0,2] == 'c'
    assert c1[0] == 'a' and c1[1] == 'b' and c1[2] == 'c'


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(Cell(shape=2), True, id="0"),
        pytest.param(ce("foo", "bar"), True, id="1"),
        pytest.param(cell(shape=2), True, id="2"),
        pytest.param(np.array([1,2]), False, id="3"),
        pytest.param([1,2], False, id="4")
    ]
)
def test_iscell(var, expected):
    assert matlab.iscell(var) == expected


@pytest.mark.parametrize(
    "var,expected",
    [
        pytest.param(struct(2), True, id="0"), # is a StructArray
        pytest.param(struct("foo", "bar"), True, id="1"),
        pytest.param(struct(foo="bar"), True, id="2"),
        pytest.param(Struct(foo="bar"), True, id="3"),
        pytest.param(struct(knots=cell((1,3))), True, id="4") # is a StructArray
    ]
)
def test_isstruct(var, expected):
    assert matlab.isstruct(var) == expected