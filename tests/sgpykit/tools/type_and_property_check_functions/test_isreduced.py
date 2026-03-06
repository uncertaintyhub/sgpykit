import pytest
import numpy as np

from sgpykit.main import create_sparse_grid, reduce_sparse_grid
from sgpykit.src import tensor_grid
from sgpykit.tools.converter_functions import asreduced
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.tools.type_and_property_check_functions import isreduced
from sgpykit.util import matlab
from sgpykit.util.misc import to_array


def test_isreduced():
    N = 2
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    lev2knots = to_array(lev2knots_doubling, N)
    i = np.array([1, 2])
    m = np.array([func(val) for func, val in zip(lev2knots, i)], dtype=np.int64)
    a_tensor_grid = tensor_grid(N, m, knots)

    # some dummy struct with same fieldnames
    dummy_reduced = matlab.struct(knots=None,
                                  size=None,
                                  weights=None,
                                  m=None,
                                  n=None)
    # some StructArray with same fieldnames
    empty_cells = matlab.cell((0, 0))
    dummy_struct_array_0 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         n=empty_cells,
                                         m=empty_cells)
    empty_cells = matlab.cell((1, 1))
    dummy_struct_array_1 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         n=empty_cells,
                                         m=empty_cells)
    empty_cells = matlab.cell((1, 2))
    dummy_struct_array_2 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         n=empty_cells,
                                         m=empty_cells)

    # a tensor grid in a sparse grid struct
    w = 2
    sparse_grid, _ = create_sparse_grid(N, w, knots, lev2knots)
    reduced_sparsegrid = reduce_sparse_grid(sparse_grid)
    assert isreduced(a_tensor_grid) == 0
    assert isreduced(dummy_reduced) == 1         # struct, so ok
    assert isreduced(dummy_struct_array_0) == 0  # StructArray, but no entries
    assert isreduced(dummy_struct_array_1) == 1  # StructArray, 1x1 cells
    assert isreduced(dummy_struct_array_2) == 0  # StructArray, 1x2 cells
    assert isreduced(sparse_grid[0]) == 0
    assert isreduced(sparse_grid) == 0
    assert isreduced(reduced_sparsegrid) == 1