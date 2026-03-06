import numpy as np

from sgpykit.src.generate_pattern import generate_pattern
from sgpykit.util import matlab
from sgpykit.util.misc import to_array


def tensor_grid(N, m, knots):
    """
    Generate a tensor grid and compute the corresponding weights.

    This function creates a tensor grid in N dimensions with M=[m1, m2, ..., mN] points
    in each direction. The knots can be specified either as a cell array of functions
    (one for each dimension) or as a single function to be used for all dimensions.

    Parameters
    ----------
    N : int
        Number of dimensions.
    m : array_like
        Number of points in each dimension.
    knots : callable or array_like of callables
        Function(s) to generate the knots in each direction. If a single function is provided,
        it is replicated for all dimensions. Each function should have the signature
        x, w = knots_function(m).

    Returns
    -------
    S : Struct
        A structure containing the tensor grid information:
        - knots : ndarray
            Vector containing the tensor grid knots.
        - weights : ndarray
            Vector containing the corresponding weights.
        - size : int
            Size of the tensor grid, equal to the product of m.
        - knots_per_dim : list
            List of N components, each containing the set of 1D knots used to build the tensor grid.
        - m : ndarray
            Input vector m, where m[i] is the length of knots_per_dim[i].
    """
    assert knots is not None
    assert N > 0

    if callable(knots):
        knots = to_array(knots, N)

    sz = np.max((1, np.prod(m, dtype=np.int64)))

    # create a "scalar" StructArray
    S = matlab.struct(
        knots=np.zeros((N, sz)),
        weights=np.ones((1, sz)),
        size=sz,
        # NOTE: do not use cell here (otherwise it becomes a non-scalar StructArray)
        knots_per_dim=np.empty(N, dtype=object),
        m=m)

    # generate the pattern that will be used for knots and weights matrices, e.g.

    # pattern = [1 1 1 1 2 2 2 2;
    #            1 1 2 2 1 1 2 2;
    #            1 2 1 2 1 2 1 2]

    # meaning "first node d-dim uses node 1 in direction 1, 2 and 3, second d-dim  node uses node 1 in
    # direction 1 and 2 and node 2 in direction 3 ...

    pattern = generate_pattern(m)
    for n in range(N):
        xx, ww = knots[n](m[n])
        S.knots_per_dim[n] = np.atleast_1d(xx).tolist()
        S.knots[n] = np.atleast_1d(xx)[pattern[n,:]]
        S.weights = S.weights * np.atleast_1d(ww)[pattern[n,:]]

    return S
