import pytest
import numpy as np

from sgpykit.src import tensor_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.tools.type_and_property_check_functions import isequal_tensor_grids
from sgpykit.util.misc import to_array


def test_isequal_tensor_grids():
    N = 2
    knots = lambda n: knots_CC(n, -1, 1, 'nonprob')
    lev2knots = to_array(lev2knots_doubling, N)
    i0 = np.array([1, 2])
    m0 = np.array([func(val) for func, val in zip(lev2knots, i0)], dtype=np.int64)
    t0 = tensor_grid(N, m0, knots)

    i1 = np.array([2, 1])
    m1 = np.array([func(val) for func, val in zip(lev2knots, i1)], dtype=np.int64)
    t1 = tensor_grid(N, m1, knots)
    equal, whatfield_eq = isequal_tensor_grids(t0, t0)
    notequal, whatfield_neq = isequal_tensor_grids(t0, t1)
    assert equal == True and whatfield_eq == ''
    assert notequal == False and whatfield_neq == 'm'