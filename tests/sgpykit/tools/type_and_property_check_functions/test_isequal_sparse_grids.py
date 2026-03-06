import pytest
import numpy as np

from sgpykit.main import create_sparse_grid
from sgpykit.tools.knots_functions import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling
from sgpykit.tools.type_and_property_check_functions import isequal_sparse_grids
from sgpykit.util.misc import to_array


def test_isequal_sparse_grids():
    N = 2
    w = 2
    knots0 = lambda n: knots_CC(n, -1, 1, 'nonprob')
    lev2knots = to_array(lev2knots_doubling, N)
    s0, _ = create_sparse_grid(N, w, knots0, lev2knots)

    knots1 = lambda n: knots_CC(n, -0.5, 0.5, 'nonprob')
    s1, _ = create_sparse_grid(N, w, knots1, lev2knots)
    equal, whatfield_eq = isequal_sparse_grids(s0, s0)
    notequal, whatfield_neq = isequal_sparse_grids(s0, s1)
    assert equal == True and whatfield_eq == ''
    assert notequal == False and whatfield_neq == 'weights'