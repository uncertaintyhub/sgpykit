import pytest
import numpy as np

from sgpykit.src import tensor_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.util.matlab import cell, ce, struct, fieldnames
from sgpykit.util.checks import is_list_math_equal
from sgpykit.util.struct import Struct


def test_StructMemberInit():
    numbers = np.array([[1, 3, 5], [2, 3, 5], [1, 1, 1], [-1, 0, 1]])
    S = Struct(knots=numbers)
    assert np.all(S.knots.shape == (4,3))
    np.testing.assert_array_almost_equal(S.knots, numbers)


def test_StructCellIndex():
    nb = 3
    s1 = struct('f1', cell((1,nb)))
    assert len(s1) == nb


def test_structMixed():
    # matlab: s1 = struct('f1',zeros(1,10), 'f2',{'a','b','c'}, 'f3',[1 2 4])
    field1 = 'f1'
    value1 = np.zeros((1, 10))
    field2 = 'f2'
    value2 = ce('a', 'b', 'c')
    field3 = 'f3'
    testa = np.array([1, 2, 4])
    value3 = testa
    # without cells
    s0 = struct(field1, value1, field3, value3)
    assert len(s0) == 1
    # with cells
    s1 = struct(field1, value1, field2, value2, field3, value3)
    assert len(s1) == 3
    # values and fieldnames match
    assert np.all([np.all(s1[i][field1] == value1) for i in range(3)])
    assert set(fieldnames(s1)) == {field1, field2, field3}
    assert np.all([s1[k][field2] == value2[k] for k in range(3)])
    assert is_list_math_equal(s1[field2], value2.tolist())
    # s1(1).('f2') = 'y'  # changes a cell element
    s1[1][field2] = 'y'
    assert s1[0][field2] == value2[0] and s1[2][field2] == value2[2]
    assert s1[1][field2] == 'y'
    # s1(1).('f1') = -1  # changes an attribute for member f1
    s1[1][field1] = -1
    assert s1[1][field1] == -1
    np.testing.assert_array_almost_equal(s1[2][field1], value1)  # second cell did not change field1
    # check that inner content has not changed, so args above have passed by copy not by reference
    test_orig = testa.copy()
    testa[0] = 99
    assert np.all([np.all(s1[i][field3] == test_orig) for i in range(3)])


def test_struct_example():
    N = 2
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    lev2knots = cell((1, N))
    for i in range(N):
        lev2knots[i] = lev2knots_doubling
    nb_grids = 1
    empty_cells = cell((1,nb_grids))
    S = struct('knots', empty_cells, 'weights', empty_cells, 'size', empty_cells, 'knots_per_dim', empty_cells,
               'm', empty_cells)
    i = np.array([1, 2])
    m = np.empty(i.shape[0], dtype=np.int64)  # 0*i
    for n in range(i.shape[0]):
        m[n] = int(lev2knots[n](i[n]))

    S[0] = tensor_grid(N, m, knots)
    #S[0].weights = S[0].weights * coeff[j] # TODO:

# def test_struct_nested():
#     s1 = struct('f1', Struct())
    pytest.skip("TODO")