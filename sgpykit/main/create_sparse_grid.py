import numpy as np

from sgpykit.src.tensor_grid import tensor_grid
from sgpykit.tools.idxset_functions.multiidx_gen import multiidx_gen
from sgpykit.util.create_sparse_grid_construct import create_sparse_grid_construct
from sgpykit.util.misc import to_array, apply_lev2knots
from sgpykit.util.struct_array import StructArray


def create_sparse_grid(N, w, knots, lev2knots, idxset=None, S2=None):
    """
    Generate a sparse grid (and corresponding quadrature weights) as a linear
    combination of full tensor grids, employing formula (2.9) of
    [Nobile-Tempone-Webster, SINUM 46/5, pages 2309-2345].

    Parameters
    ----------
    N : int
        Number of dimensions.
    w : int
        Integer non-negative value defining the multiindex set.
    knots : callable or list of callables
        Function(s) to generate knots in each direction. If a single function,
        it is used in every direction. The header of knots_function is
        [x,w]=knots_function(m).
    lev2knots : callable or list of callables
        Function(s) defining the relation between level and number of knots.
        If a single function, it is used in every direction. The header of
        m_function is m=m_function(i).
    idxset : callable, optional
        Function defining the multiindex set. Default is sum(i-1).
    S2 : StructArray, optional
        Another sparse grid to recycle tensor grids from.

    Returns
    -------
    S : StructArray
        Structure containing the information on the sparse grid (vector of
        tensor grids). Each tensor grid S[j] is a structure with the following
        fields:
            knots : ndarray
                Vector containing the tensor grid knots.
            weights : ndarray
                Vector containing the corresponding weights.
            size : int
                Size of the tensor grid, S[j].size = prod(m).
            knots_per_dim : list
                Cell array (N components), each component is the set of 1D knots
                used to build the tensor grid.
            m : ndarray
                The input vector m, m == lev2knots(idx), m(i)==length(S[j].knots_per_dim[i]).
            coeff : int
                How many times the tensor grid appears in the sparse grid (with sign).
            idx : ndarray
                The multiidx vector corresponding to the current grid.
    C : ndarray
        Multi-index set used to generate the sparse grid. It is sorted
        lexicographically and contains all multi-indices, even those whose
        coefficient in the combination technique is 0.

    Notes
    -----
    The reduced multi-idx set D with only the indices with non-zero coefficient
    can be obtained as:
    ```
    nb_idx = len(S)
    D = np.vstack(S.idx).T # D.shape is (N, nb_idx)
    ```
    and will be lexicographic sorted as well.
    """
    # input handling
    if not idxset:
        idxset = lambda i: sum(i) # i in 0-based indexing case

    # if knots and lev2knots are simple function, we replicate them in a cell
    if callable(knots):
        knots = to_array(knots, N)

    if callable(lev2knots):
        lev2knots = to_array(lev2knots, N)

    if w == 0: # TODO: test this case
        # the trivial case
        i = np.zeros(N)
        m = apply_lev2knots(i, lev2knots, N) # TODO: index check
        S = StructArray(1)
        S[0] = tensor_grid(N, m, knots)
        S[0].coeff = 1
        S[0].idx = i
        C = i
        return S, C

    # let's go with the sparse construction
    # build the list of multiindices in the set: idxset(i)<=w.
    C = multiidx_gen(N=N, rule=idxset, w=w, base=0)
    S, C = create_sparse_grid_construct(C, N, knots, lev2knots, S2, base=0)

    return S, C
