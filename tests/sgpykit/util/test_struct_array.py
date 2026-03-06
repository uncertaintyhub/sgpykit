import pytest
import numpy as np

from sgpykit.util.matlab import cell, struct, fieldnames
from sgpykit.util.struct_array import StructArray


def test_StructArray1D():
    nb = 4
    shapes_1d = ((1, nb), (nb, 1))
    for shape in shapes_1d:
        empty_cells = cell(shape)
        S = StructArray(knots=empty_cells,
                        number=empty_cells)
        val = 12
        for i in range(nb):
            S[i].knots = i
            S[i].number = val
        check = True
        for i in range(nb):
            if S[i].knots != i or S[i].number != val:
                check = False
                break
        assert check


def test_StructArray2D():
    nrows, ncols = 3, 4
    shape = (nrows, ncols)
    empty_cells = cell(shape)
    S = StructArray(knots=empty_cells,
                    number=empty_cells)
    val = 12
    for i in range(nrows):
        for j in range(ncols):
            S[i][j].knots = i+j*nrows
            S[i][j].number = val
    check = True
    for i in range(nrows):
        for j in range(ncols):
            if S[i][j].knots != (i+j*nrows) or S[i][j].number != val:
                check = False
                break
        if check == False:
            break
    assert check


def test_StructArrayFieldnamesDynamic():
    values1 = [3, 9, 12, -1]
    values2 = [-3.0, 5.3, 9.0, 95]
    ncells = len(values1)
    empty_cells = cell((1, ncells))
    s1 = struct(values1=empty_cells,
                values2=empty_cells)
    fields = fieldnames(s1)
    for i in range(ncells):
        s1[i].values1 = values1[i]
        s1[i].values2 = values2[i]
    s2 = struct(values1=cell((1, ncells)),
                values2=cell((1, ncells)))
    for i in range(ncells):
        for fn in fields:
            s2[i][fn] = s1[i][fn]
    assert np.all([s2[i]['values1'] == values1[i] for i in range(ncells)])
    assert np.all([s2[i]['values2'] == values2[i] for i in range(ncells)])


def test_structarray_assign_struct():
    N = 2
    sz = 3
    nb_grids = 2 # TODO: <N ?
    empty_cells = cell((1,nb_grids))
    s0 = struct(knots=np.zeros((N, sz)),
                weights=np.ones((1, sz)),
                size=sz,
                knots_per_dim=cell((1, N)))
    S = struct('knots', empty_cells, 'weights', empty_cells, 'size', empty_cells, 'knots_per_dim', empty_cells)
    S[0] = s0
    pytest.skip("TODO")
