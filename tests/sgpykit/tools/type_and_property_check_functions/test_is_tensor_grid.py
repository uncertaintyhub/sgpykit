import pytest
import numpy as np

from sgpykit.main import create_sparse_grid
from sgpykit.src import tensor_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.tools.type_and_property_check_functions import is_tensor_grid
from sgpykit.util import matlab
from sgpykit.util.misc import to_array


def test_is_tensor_grid():
    N = 2
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    lev2knots = to_array(lev2knots_doubling, N)
    i = np.array([1, 2])
    m = np.array([func(val) for func, val in zip(lev2knots, i)], dtype=np.int64)
    a_tensor_grid = tensor_grid(N, m, knots)

    # some dummy struct with same fieldnames  # TODO: is it a grid if entries are none?
    dummy_tensor_grid = matlab.struct(knots=None,
                                      size=None,
                                      weights=None,
                                      knots_per_dim=None,
                                      m=None)
    # some StructArray with same fieldnames
    empty_cells = matlab.cell((0, 0))
    dummy_struct_array_0 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         knots_per_dim=empty_cells,
                                         m=empty_cells)
    empty_cells = matlab.cell((1, 1))
    dummy_struct_array_1 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         knots_per_dim=empty_cells,
                                         m=empty_cells)
    empty_cells = matlab.cell((1, 2))
    dummy_struct_array_2 = matlab.struct(knots=empty_cells,
                                         size=empty_cells,
                                         weights=empty_cells,
                                         knots_per_dim=empty_cells,
                                         m=empty_cells)

    # a tensor grid in a sparse grid struct
    w = 2
    sparse_grid, _ = create_sparse_grid(N, w, knots, lev2knots)
    # not a tensor grid
    no_tensor_grid = matlab.struct(knots=empty_cells)
    assert is_tensor_grid(a_tensor_grid) == 1
    assert is_tensor_grid(dummy_tensor_grid) == 1    # struct, so ok
    assert is_tensor_grid(dummy_struct_array_0) == 0 # StructArray, but no entries
    assert is_tensor_grid(dummy_struct_array_1) == 1 # StructArray, 1x1 cells
    assert is_tensor_grid(dummy_struct_array_2) == 0 # StructArray, 1x2 cells
    assert is_tensor_grid(sparse_grid[0]) == -1
    assert is_tensor_grid(no_tensor_grid) == 0