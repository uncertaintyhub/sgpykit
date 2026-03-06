from sgpykit.main.create_sparse_grid import create_sparse_grid
from sgpykit.main.reduce_sparse_grid import reduce_sparse_grid
from sgpykit.tools.knots_functions.knots_CC import knots_CC
from sgpykit.tools.lev2knots_functions import lev2knots_doubling


def create_sparse_grid_quick_preset(N, w):
    """
    Create a sparse grid with a quick preset configuration.

    This function creates a "vanilla" sparse grid using Clenshaw-Curtis points
    in the interval [-1, 1] with the lev2knots_doubling rule and a multi-index
    set defined by sum(ii) <= w. It is a shortcut for the following operations:

    .. code-block:: python

        knots = lambda n: knots_CC(n, -1, 1)
        S, C = create_sparse_grid(N, w, knots, lev2knots_doubling)
        Sr = reduce_sparse_grid(S)

    Parameters
    ----------
    N : int
        Number of dimensions.
    w : int
        Level of the sparse grid.

    Returns
    -------
    S : struct
        The full sparse grid structure.
    Sr : struct
        The reduced sparse grid structure.
    C : ndarray
        Coefficient matrix for the combination technique.

    Notes
    -----
    This function is equivalent to using the 'SM' (Smolyak) rule in
    define_functions_for_rule.
    """
    knots = lambda n: knots_CC(n, -1, 1)
    S, C = create_sparse_grid(N, w, knots, lev2knots_doubling)
    Sr = reduce_sparse_grid(S)
    return S, Sr, C
